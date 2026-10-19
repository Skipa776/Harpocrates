import type { Hook, Register } from 'claude-code'

import {
  ARGUMENT_HINT,
  BAD_ANSWER,
  CANCELLED,
  CANCEL_LABEL,
  CONFIRM_LABEL,
  HELP,
  LIST_ARGV,
  NOT_FOUND,
  NOT_INSTALLED,
  REFUSED,
  absolutePath,
  changeArgv,
  cliCommand,
  confirmQuestion,
  doneText,
  failText,
  listText,
  parseAllowArgs,
  readAnswer,
  recordPaths,
  recordedInterpreter,
  unknownText,
  verbOf,
} from './allowlist'
import type { Change, CliAnswer, Interpreter, Target } from './allowlist'
import {
  ALLOW_EDITS,
  HEADER,
  KEEP_PROTECTED,
  denyReason,
  EDIT_TOOLS,
  isHarpocratesReason,
  isSameFile,
  noAnswerReason,
  parseReason,
  questionFor,
  sanitize,
  targetPath,
  uniqueQuestion,
} from './parse'
import type { CallLine, ParsedAsk } from './parse'
import { HINT, NOTE, SUBAGENT, SUBTITLE, TITLE, buildCard } from './layout'

type Pending = {
  readonly ask: ParsedAsk
  readonly call: CallLine
  readonly isSubagent: boolean
}

// Question text -> the ask waiting under it. Every pending question is unique,
// so the card is always drawn from its own call. After a reload this is empty
// and the engine draws its own dialog.
const pending = new Map<string, Pending>()

type AskedQuestion = { question: string; header: string; options: readonly unknown[] }

/** The dialog's first question, when it has the shape the engine draws. */
function askedQuestion(questions: readonly unknown[]): AskedQuestion | undefined {
  const q = questions[0]
  if (typeof q !== 'object' || q === null) return undefined
  const { question, header, options } = q as { question?: unknown; header?: unknown; options?: unknown }
  return typeof question === 'string' && typeof header === 'string' && Array.isArray(options)
    ? (q as AskedQuestion)
    : undefined
}

const DESCRIPTIONS: Readonly<Record<string, string>> = {
  [KEEP_PROTECTED]: 'Claude keeps using safe_read; the file stays unedited',
  [ALLOW_EDITS]: 'Built-in Read and Edit of this file, until the session ends',
}

const withDescription = (option: unknown): unknown => {
  if (typeof option !== 'object' || option === null) return option
  const label = (option as { label?: unknown }).label
  const description = typeof label === 'string' ? DESCRIPTIONS[label] : undefined
  return description === undefined ? option : { ...option, description }
}

type CommandApi = Parameters<Hook<'command.run'>>[0]

/** What the CLI answered; a string instead is the text to show because it could not be run. */
type CliRun = { readonly exitCode: number; readonly answer: CliAnswer | null }
type Outcome = CliRun | string

/**
 * The interpreter `harpocrates install` recorded, read from the person's
 * install record; null when there is no record, no usable "python" in it, or
 * it cannot be read. The first record file that exists decides (the new state
 * folder before the old one, so a stale old record never overrides it). There
 * is no other fallback: a `harpocrates` found on PATH (or a project-local
 * shim) is never run.
 */
async function readInterpreter($: CommandApi): Promise<Interpreter | null> {
  try {
    const home = (await $.env.get('HOME')) ?? (await $.env.get('USERPROFILE'))
    if (!home) return null
    for (const path of recordPaths(home)) {
      let text: unknown
      try {
        text = await $.fs.read(path)
      } catch {
        continue // no record in this folder: look in the next
      }
      return typeof text === 'string' ? recordedInterpreter(text) : null
    }
    return null
  } catch {
    return null
  }
}

/** Runs the CLI as `<recorded python> <recorded flags> -m Harpocrates.cli ...` by argv (no shell, so typed text is never parsed by one). */
async function runCli($: CommandApi, argv: readonly string[]): Promise<Outcome> {
  const interpreter = await readInterpreter($)
  if (interpreter === null) return NOT_INSTALLED

  try {
    const ran = await $.process.run(cliCommand(interpreter, argv))
    return { exitCode: ran.exitCode, answer: readAnswer(ran.stdout) }
  } catch {
    return NOT_FOUND
  }
}

/** The text for a finished change; never the CLI's raw output. */
function changeResult(kind: Change, target: Target, run: Outcome): string {
  if (typeof run === 'string') return run
  if (run.answer === null) return BAD_ANSWER
  if (run.exitCode === 0 && run.answer.isOk) return doneText(kind, target, run.answer)
  return failText(kind, run.answer.error ?? `the CLI exited with code ${run.exitCode}`)
}

/** A path made absolute from the session's folder and home; undefined when a base it needs is unknown. */
async function absoluteFor($: CommandApi, path: string): Promise<string | undefined> {
  let cwd: string | undefined
  let home: string | undefined
  try {
    cwd = await $.session.cwd()
  } catch {
    cwd = undefined
  }
  if (path === '~' || path.startsWith('~/')) {
    try {
      home = (await $.env.get('HOME')) ?? (await $.env.get('USERPROFILE'))
    } catch {
      home = undefined
    }
  }

  return absolutePath(path, cwd, home) ?? undefined
}

async function change($: CommandApi, kind: Change, target: Target): Promise<string> {
  let resolved = target
  if (target.kind === 'path') {
    const path = await absoluteFor($, target.value)
    if (path === undefined) return failText(kind, 'give a full path (the session folder or home folder is unknown)')
    resolved = { kind: 'path', value: path }
  }

  const run = await runCli($, changeArgv(kind, resolved))
  // Exit 3: a broad name. Only the person's exact "Allow it everywhere" re-runs
  // with --yes; every other answer, a dismissal included, cancels.
  if (kind !== 'add' || resolved.kind !== 'name' || typeof run === 'string' || run.exitCode !== 3) {
    return changeResult(kind, resolved, run)
  }

  let choice: string | undefined
  try {
    choice = await $.ui.ask(confirmQuestion(resolved.value), [CONFIRM_LABEL, CANCEL_LABEL])
  } catch {
    choice = undefined
  }

  return choice === CONFIRM_LABEL
    ? changeResult(kind, resolved, await runCli($, changeArgv(kind, resolved, true)))
    : CANCELLED
}

async function list($: CommandApi): Promise<string> {
  const run = await runCli($, LIST_ARGV)
  if (typeof run === 'string') return run
  if (run.exitCode !== 0 || run.answer?.entries == null) return run.answer?.error ?? BAD_ANSWER
  return listText(run.answer.entries)
}

export const register: Register = on => {
  on('session.start', async ($, e, next) => {
    try {
      await $.command.register({
        name: 'harpocrates',
        description: 'Manage the Harpocrates allowlist',
        argumentHint: ARGUMENT_HINT,
      })
    } catch {
      $.ui.log('harpocrates-prompt: could not register /harpocrates', { to: 'debug' })
    }

    return next(e)
  })

  // Only the person's own Enter in the chat box (origin "composer") changes the
  // allowlist. Anything else (a plugin, the SDK, the bridge, a model-invoked
  // run, an unclassified origin) is refused before anything is parsed further
  // or run. Help and the listing change nothing, so any origin may run them.
  // The answer is `text` only: nothing is put in `context` for the model.
  on('command.run', { command: 'harpocrates' }, async ($, e) => {
    try {
      const cmd = parseAllowArgs(e.args)
      const word = verbOf(e.args)
      const isChange = (word === 'allow' || word === 'remove' || cmd.kind === 'add' || cmd.kind === 'remove') && cmd.kind !== 'list'

      if (isChange && e.origin?.kind !== 'composer') return { text: REFUSED }

      switch (cmd.kind) {
        case 'help':
          return { text: HELP }
        case 'unknown':
          return { text: unknownText(cmd.word) }
        case 'usage':
          return { text: cmd.message }
        case 'list':
          return { text: await list($) }
        default:
          return { text: await change($, cmd.kind, cmd.target) }
      }
    } catch {
      $.ui.log('harpocrates-prompt: /harpocrates failed', { to: 'debug' })
      return { text: 'The /harpocrates command failed. Type /harpocrates allowlist to see the current list.' }
    }
  })

  // The person's decision comes from $.ui.ask, a `$` call: the hook's 10 s
  // budget does not run while it is in flight. Only an exact "Allow edits this
  // session" allows; every other answer, free text and a dismissal included,
  // denies. Anything we cannot read, and every query, is left to the engine's
  // dialog.
  on('tool.check', async ($, e, next) => {
    const v = await next(e)

    if (
      !e.tool_use_id ||
      v.decision !== 'ask' ||
      v.hook !== 'PreToolUse' ||
      v.rule !== undefined ||
      !isHarpocratesReason(v.reason) ||
      !EDIT_TOOLS.includes(e.tool)
    ) {
      return v
    }

    // The card must describe the real call: the call's own file has to be the
    // one the reason names, or the engine's dialog is left to say it.
    const target = targetPath(e.input)
    const parsed = parseReason(v.reason)

    if (target === undefined || !parsed) return v

    let cwd: string | undefined
    try {
      cwd = await $.session.cwd()
    } catch {
      cwd = undefined
    }

    if (!isSameFile(parsed.path, target, cwd)) return v

    const ask: ParsedAsk = { path: sanitize(target), kinds: parsed.kinds }

    const question = uniqueQuestion(questionFor(ask), q => pending.has(q))

    pending.set(question, {
      ask,
      call: { label: e.tool, path: ask.path },
      isSubagent: !!e.agentId,
    })

    try {
      const answer = await $.ui.ask(question, { header: HEADER, options: [KEEP_PROTECTED, ALLOW_EDITS] })

      return answer === ALLOW_EDITS
        ? { decision: 'allow' as const }
        : { decision: 'deny' as const, reason: denyReason(ask) }
    } catch {
      $.ui.log(`harpocrates-prompt: question not answered (${e.tool})`, { to: 'debug' })
      return { decision: 'deny' as const, reason: noAnswerReason(ask) }
    } finally {
      pending.delete(question)
    }
  }).catch(($, e, next) => next(e))

  on('ui.render', { component: 'AskUserQuestion' }, async ($, e, next) => {
    const first = askedQuestion(e.props.questions)
    const entry = first?.header === HEADER ? pending.get(first.question) : undefined

    if (!first || !entry) {
      return next(e)
    }

    const { ask, call, isSubagent } = entry
    const dialog = await next({
      ...e,
      props: {
        ...e.props,
        questions: [
          { ...first, options: first.options.map(withDescription) },
          ...e.props.questions.slice(1),
        ],
      },
    })

    const { Box, Text } = $.ui.resolve(e)
    const columns = e.viewport?.columns ?? 80
    const card = buildCard({ ask, call, isSubagent, columns })

    return (
      <Box flexDirection="column">
        <Box borderStyle="round" borderColor="warning" flexDirection="column" paddingX={1}>
          <Text>
            <Text bold>{TITLE}</Text>
            <Text dimColor>{SUBTITLE}</Text>
            {isSubagent ? <Text dimColor>{SUBAGENT}</Text> : null}
          </Text>
          <Text wrap="truncate-end">
            <Text dimColor>{card.call.label}</Text>
            {card.call.rest}
          </Text>
          <Text>{card.body}</Text>
          <Text>{NOTE}</Text>
          {card.hasHint ? <Text dimColor>{HINT}</Text> : null}
        </Box>
        {dialog}
      </Box>
    )
  })
}

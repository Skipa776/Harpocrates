import type { Register } from 'claude-code'

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

export const register: Register = on => {
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

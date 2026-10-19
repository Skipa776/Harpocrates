// /harpocrates: parsing and wording for the allowlist command. Pure: nothing
// here runs a process or touches a file. The Harpocrates CLI (`allow --json`)
// is the only thing that changes the allowlist, and register.tsx is the only
// caller, and only for the person's own Enter.

import { sanitize } from './parse'
import { truncateRight } from './width'

export const HELP = [
  'Harpocrates allowlist (only you can change it; every use is logged)',
  '/harpocrates allow <path>            allow one file or folder',
  '/harpocrates allow --name <name>     allow a folder or file name in every project',
  '/harpocrates remove <path>           remove a path',
  '/harpocrates remove --name <name>    remove a name',
  '/harpocrates allowlist               show the list',
].join('\n')

export const ARGUMENT_HINT = 'allow <path> | allow --name <name> | remove … | allowlist'
export const REFUSED = 'Only you can change the Harpocrates allowlist: type /harpocrates in the chat box.'
export const NOT_FOUND = 'Harpocrates CLI not found. Install it: pip install git+https://github.com/Skipa776/harpocrates'
export const NOT_INSTALLED =
  'Run `harpocrates install` first: the plugin only runs the Harpocrates interpreter that install recorded.'
export const BAD_ANSWER ='The Harpocrates CLI returned an answer this plugin cannot read.'
export const CONFIRM_LABEL = 'Allow it everywhere'
export const CANCEL_LABEL = 'Cancel'
export const CANCELLED = 'Cancelled.'

export type Target = { readonly kind: 'path' | 'name'; readonly value: string }
export type Change = 'add' | 'remove'

export type AllowCommand =
  | { readonly kind: 'help' }
  | { readonly kind: 'list' }
  | { readonly kind: Change; readonly target: Target }
  | { readonly kind: 'unknown'; readonly word: string }
  | { readonly kind: 'usage'; readonly message: string }

/**
 * Splits on whitespace. "Double" and 'single' quotes group a word (and are
 * dropped), so a path with spaces is one word. There are no escapes, so a
 * Windows backslash stays a backslash. Null when a quote is never closed.
 */
export function tokenize(text: string): string[] | null {
  const words: string[] = []
  let word = ''
  let isOpen = false
  let quote: string | null = null

  for (const ch of text) {
    if (quote !== null) {
      if (ch === quote) quote = null
      else word += ch
    } else if (ch === '"' || ch === "'") {
      quote = ch
      isOpen = true
    } else if (/\s/.test(ch)) {
      if (isOpen) words.push(word)
      word = ''
      isOpen = false
    } else {
      word += ch
      isOpen = true
    }
  }

  if (quote !== null) return null
  if (isOpen) words.push(word)
  return words
}

/** The command word as parsing sees it (quotes dropped, lower-cased), for the origin check. */
export const verbOf = (args: string): string => tokenize(args)?.[0]?.toLowerCase() ?? ''

const CONTROL =/[\u0000-\u001f\u007f]/

function parseTarget(kind: Change, rest: readonly string[]): AllowCommand {
  const verb = kind === 'add' ? 'allow' : 'remove'
  const usage = (message: string): AllowCommand => ({ kind: 'usage', message })
  const make = (target: Target): AllowCommand => ({ kind, target })

  if (rest.length === 0) return usage(`Usage: /harpocrates ${verb} <path>  or  /harpocrates ${verb} --name <name>`)
  if (rest.some(w => CONTROL.test(w))) return usage('A path or name cannot hold control characters.')

  if (rest[0] === '--name') {
    const name = rest[1]
    if (rest.length !== 2 || name === undefined || name === '') return usage(`Usage: /harpocrates ${verb} --name <name>`)
    if (name.startsWith('-')) return usage(`Unknown option: ${truncateRight(sanitize(name), 40)}`)
    return make({ kind: 'name', value: name })
  }

  const path = rest[0]
  if (path === undefined || path === '') return usage(`Usage: /harpocrates ${verb} <path>`)
  if (path.startsWith('-')) {
    return usage(`Unknown option: ${truncateRight(sanitize(path), 40)} (write ./ before a path that starts with a dash)`)
  }
  if (rest.length > 1) return usage(`Put a path that contains spaces in quotes: /harpocrates ${verb} "my folder/file"`)
  return make({ kind: 'path', value: path })
}

/** What the person typed after `/harpocrates`. */
export function parseAllowArgs(args: string): AllowCommand {
  const words = tokenize(args)
  if (words === null) return { kind: 'usage', message: 'A quote is never closed.' }

  const [verb, ...rest] = words
  if (verb === undefined) return { kind: 'help' }

  switch (verb.toLowerCase()) {
    case 'help':
      return { kind: 'help' }
    case 'allowlist':
    case 'list':
      return { kind: 'list' }
    case 'allow':
      return rest.length === 1 && rest[0] === '--list' ? { kind: 'list' } : parseTarget('add', rest)
    case 'remove':
      return parseTarget('remove', rest)
    default:
      return { kind: 'unknown', word: truncateRight(sanitize(verb), 40) }
  }
}

export const unknownText = (word: string): string => `Unknown: ${word}\n${HELP}`

const isAbsolute = (path: string): boolean => path.startsWith('/') || /^[A-Za-z]:[\\/]/.test(path)
const trimEnd = (dir: string): string => dir.replace(/[\\/]+$/, '')

/**
 * `path` made absolute: a leading `~` expanded with `home`, a relative path
 * joined to `cwd`. Dots are left for the CLI, which resolves symlinks. Null
 * when the base it needs is unknown.
 */
export function absolutePath(path: string, cwd: string | undefined, home: string | undefined): string | null {
  if (path === '~' || path.startsWith('~/')) return home ? `${trimEnd(home)}${path.slice(1)}` : null
  if (isAbsolute(path)) return path
  return cwd ? `${trimEnd(cwd)}/${path}` : null
}

/**
 * Where `install` records the interpreter that runs the hooks, in order: the
 * state folder `.harpo` (FR-NAME-04), then `.harpocrates`, which is where the
 * record sits until the move happens (or when it was refused).
 */
export const recordPaths = (home: string): readonly string[] => [
  `${trimEnd(home)}/.harpo/install.json`,
  `${trimEnd(home)}/.harpocrates/install.json`,
]

export type Interpreter = { readonly python: string; readonly args: readonly string[] }

/**
 * `python -m` puts the working directory (the agent-controlled project) first
 * on sys.path, so a planted Harpocrates package there would run instead of the
 * real one. Only these flags, which drop it, are accepted from the record:
 * -P (3.11 and later) and -I (isolated mode).
 */
const SAFE_FLAGS: readonly string[] = ['-I', '-P']

/**
 * The interpreter the install record names (`"python"`) and the flags it runs
 * with (`"python_args"`, `["-I"]` when the record has none). The path must be
 * absolute with no control characters; the flags must be a non-empty list of
 * -I or -P. Null for anything else (no such key, a relative or bare name,
 * other flags, garbage). The plugin never falls back to PATH.
 */
export function recordedInterpreter(text: string): Interpreter | null {
  let raw: unknown
  try {
    raw = JSON.parse(text)
  } catch {
    return null
  }
  if (typeof raw !== 'object' || raw === null || Array.isArray(raw)) return null

  const { python, python_args: flags } = raw as { python?: unknown; python_args?: unknown }
  if (typeof python !== 'string' || python === '' || !isAbsolute(python) || CONTROL.test(python)) return null
  if (flags === undefined) return { python, args: ['-I'] }
  if (!Array.isArray(flags) || flags.length === 0) return null
  if (!flags.every((f: unknown) => typeof f === 'string' && SAFE_FLAGS.includes(f))) return null

  return { python, args: flags as string[] }
}

/** The whole command: the recorded interpreter and its flags, then the CLI module. */
export const cliCommand = (interpreter: Interpreter, argv: readonly string[]): string[] => [
  interpreter.python,
  ...interpreter.args,
  '-m',
  'Harpocrates.cli',
  ...argv,
]

/** The CLI's arguments for a change; `--yes` only once the person confirmed a broad name. */
export function changeArgv(kind: Change, target: Target, isConfirmed = false): string[] {
  return [
    'allow',
    ...(kind === 'remove' ? ['--remove'] : []),
    '--json',
    ...(isConfirmed ? ['--yes'] : []),
    ...(target.kind === 'name' ? ['--name', target.value] : [target.value]),
  ]
}

export const LIST_ARGV: readonly string[] = ['allow', '--list', '--json']

export type Entry = { readonly kind: 'path' | 'name'; readonly value: string }

export type CliAnswer = {
  readonly isOk: boolean
  readonly action?: string
  readonly value?: string
  readonly error?: string
  readonly entries: readonly Entry[] | null
}

/** The CLI's `--json` line, or null when it is not an object. Only the fields the plugin reads are kept. */
export function readAnswer(stdout: string): CliAnswer | null {
  let raw: unknown
  try {
    raw = JSON.parse(stdout)
  } catch {
    return null
  }
  if (typeof raw !== 'object' || raw === null || Array.isArray(raw)) return null

  const o = raw as { ok?: unknown; action?: unknown; value?: unknown; error?: unknown; entries?: unknown }
  const entries = Array.isArray(o.entries)
    ? o.entries.flatMap((x: unknown): Entry[] => {
        const e = x as { kind?: unknown; value?: unknown } | null
        return e && (e.kind === 'path' || e.kind === 'name') && typeof e.value === 'string'
          ? [{ kind: e.kind, value: e.value }]
          : []
      })
    : null

  return {
    isOk: o.ok === true,
    action: typeof o.action === 'string' ? o.action : undefined,
    value: typeof o.value === 'string' ? o.value : undefined,
    error: typeof o.error === 'string' ? truncateRight(sanitize(o.error), 300) : undefined,
    entries,
  }
}

export function listText(entries: readonly Entry[]): string {
  const lines = (kind: Entry['kind']): string[] =>
    entries.filter(e => e.kind === kind).map(e => `  ${sanitize(e.value)}`)
  const paths = lines('path')
  const names = lines('name')
  const parts = [
    ...(paths.length > 0 ? ['Paths:', ...paths] : []),
    ...(names.length > 0 ? ['Names (every project):', ...names] : []),
  ]

  return parts.length > 0 ? parts.join('\n') : 'The allowlist is empty.'
}

/** The line the person reads after a change succeeded. */
export function doneText(kind: Change, target: Target, answer: CliAnswer): string {
  const shown = sanitize(answer.value ?? target.value)
  if (kind === 'remove') return target.kind === 'name' ? `Removed the name ${shown}.` : `Removed ${shown}.`
  if (target.kind === 'name') {
    return answer.action === 'exists'
      ? `The name ${shown} was already allowed in every project.`
      : `Allowed the name ${shown} in every project.`
  }
  return answer.action === 'exists' ? `${shown} was already allowed.` : `Allowed ${shown}.`
}

export const failText = (kind: Change, error: string): string =>
  `${kind === 'add' ? 'Not allowed' : 'Not removed'}: ${error}`

export const confirmQuestion = (name: string): string =>
  `Allow the name ${truncateRight(sanitize(name), 60)} in every project? It is broad.`

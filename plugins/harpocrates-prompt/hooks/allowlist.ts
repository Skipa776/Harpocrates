// /harpo: parsing and wording for the allowlist command (/harpocrates, its old
// name, does the same: FR-NAME-02). Pure: nothing here runs a process or
// touches a file. The Harpo CLI (`allow --json`) is the only thing that
// changes the allowlist, and register.tsx is the only caller, and only for the
// person's own Enter.

import { sanitize } from './parse'
import { truncateRight } from './width'

export const HELP = [
  'Harpo allowlist (only you can change it; every use is logged)',
  '/harpo allow <path>            allow one file or folder',
  '/harpo allow --name <name>     allow a folder or file name in every project',
  '/harpo remove <path>           remove a path',
  '/harpo remove --name <name>    remove a name',
  '/harpo allowlist               show the list',
].join('\n')

export const ARGUMENT_HINT = 'allow <path> | allow --name <name> | remove … | allowlist'
export const REFUSED = 'Only you can change the Harpo allowlist: type /harpo in the chat box.'
export const NOT_FOUND = 'Harpo CLI not found. Install it: pip install git+https://github.com/Skipa776/harpocrates'
export const NOT_INSTALLED =
  'Run `harpo install` first: the plugin only runs the Harpo interpreter that install recorded.'
export const BAD_ANSWER ='The Harpo CLI returned an answer this plugin cannot read.'
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

  if (rest.length === 0) return usage(`Usage: /harpo ${verb} <path>  or  /harpo ${verb} --name <name>`)
  if (rest.some(w => CONTROL.test(w))) return usage('A path or name cannot hold control characters.')

  if (rest[0] === '--name') {
    const name = rest[1]
    if (rest.length !== 2 || name === undefined || name === '') return usage(`Usage: /harpo ${verb} --name <name>`)
    if (name.startsWith('-')) return usage(`Unknown option: ${truncateRight(sanitize(name), 40)}`)
    return make({ kind: 'name', value: name })
  }

  const path = rest[0]
  if (path === undefined || path === '') return usage(`Usage: /harpo ${verb} <path>`)
  if (path.startsWith('-')) {
    return usage(`Unknown option: ${truncateRight(sanitize(path), 40)} (write ./ before a path that starts with a dash)`)
  }
  if (rest.length > 1) return usage(`Put a path that contains spaces in quotes: /harpo ${verb} "my folder/file"`)
  return make({ kind: 'path', value: path })
}

/** What the person typed after `/harpo` (or `/harpocrates`). */
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

export const isAbsolute = (path: string): boolean => path.startsWith('/') || /^[A-Za-z]:[\\/]/.test(path)
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
 * The state folders in the order they are tried: `.harpo` (FR-NAME-04), then
 * `.harpocrates`, which is where the record sits until the move happens (or
 * when it was refused). `.harpocrates` is looked at only when `.harpo` is not
 * there at all (SEC-09): once `.harpo` exists in any form, it decides.
 */
export const stateFolders = (home: string): readonly string[] => [
  `${trimEnd(home)}/.harpo`,
  `${trimEnd(home)}/.harpocrates`,
]

/** The install record inside a state folder. */
export const recordIn = (folder: string): string => `${folder}/install.json`

/** Where `install` records the interpreter that runs the hooks, one path per state folder. */
export const recordPaths = (home: string): readonly string[] => stateFolders(home).map(recordIn)

/** What `$.fs.stat` says about a path: what it leads to, and whether it is itself a symbolic link. */
export type PathInfo = { readonly kind: 'file' | 'dir' | 'other'; readonly isLink: boolean }

/**
 * Whether a state folder and its record may be read: the folder must be a
 * plain directory and the record a plain file, neither a symbolic link
 * (SEC-09). The record is the one thing the plugin runs a program from, so it
 * is never read through a link an agent could have planted.
 */
export const isPlainFolder = (info: PathInfo): boolean => info.kind === 'dir' && !info.isLink
export const isPlainFile = (info: PathInfo): boolean => info.kind === 'file' && !info.isLink

export type Interpreter = { readonly python: string; readonly args: readonly string[] }

/**
 * `python -m` puts the working directory (the agent-controlled project) first
 * on sys.path, so a planted Harpo package there would run instead of the
 * real one, and so would one on PYTHONPATH (SEC-08). Only these flags are
 * accepted from the record: -I (isolated mode), -P (3.11 and later: drops the
 * working directory) and -E (ignores PYTHON* variables, written with -P for an
 * install in the user site). -E alone leaves the working directory on sys.path,
 * so a list must hold -I or -P.
 */
const SAFE_FLAGS: readonly string[] = ['-E', '-I', '-P']
const DROPS_CWD: readonly string[] = ['-I', '-P']

/**
 * The interpreter the install record names (`"python"`) and the flags it runs
 * with (`"python_args"`, `["-I"]` when the record has none). The path must be
 * absolute with no control characters; the flags must be a non-empty list of
 * -I, -P or -E that holds -I or -P. Null for anything else (no such key, a
 * relative or bare name, other flags, garbage). The plugin never falls back to
 * PATH.
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
  if (!flags.some(f => DROPS_CWD.includes(f as string))) return null

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

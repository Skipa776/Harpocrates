// Reads the reason of a Harpo PreToolUse "ask" verdict (FR-READ-06: only
// edits of a file that holds secrets ask, one file per call). Only the path and
// the secret kinds are in it; secret values never are, and nothing here logs
// anything.

import { truncateLeft } from './width'

export const PREFIX = 'Harpo: '
/** What a hook from before the rename wrote ("Harpocrates: ..."): still drawn, so a plugin update needs no CLI update. */
export const LEGACY_PREFIX = 'Harpocrates: '
export const HEADER = 'Harpo'
export const KEEP_PROTECTED = 'Keep secrets protected'
export const ALLOW_EDITS = 'Allow edits this session'

/** The tools whose edits the hook asks about, and so the only ones the card is drawn for. */
export const EDIT_TOOLS: readonly string[] = ['Edit', 'Write', 'MultiEdit', 'NotebookEdit']

export type ParsedAsk = {
  readonly path: string
  /** A comma list of kinds, or "unscanned (over the size limit)". */
  readonly kinds: string
}

const KINDS = '(?:[a-z0-9_]+(?:, [a-z0-9_]+)*|unscanned \\(over the size limit\\))'
// The whole template, end to end. The path is whatever lies between the two
// anchors, so it may hold spaces and parentheses; `.+` is greedy, so a path
// that itself spells an anchor still splits at the last one, where the kinds
// are. Anything the hook words differently is not ours to draw.
const REASON = new RegExp(
  '^(?:Harpo|Harpocrates): Claude wants to edit (.+) without secret protection\\. ' +
    `It contains secrets \\((${KINDS})\\)\\. ` +
    'Allowing lets Claude read and edit this file with its built-in tools for the rest of this session; ' +
    'these values go to the model provider as-is\\. ' +
    'Choose No to keep them protected: Claude can still read it with mcp__(?:harpo|harpocrates)__safe_read\\.$',
  's',
)
// Control characters, bidi controls (which can reorder a name so ".exe" reads
// as something else), zero-width characters and line/paragraph separators.
const UNSAFE = new RegExp(
  '[\\u0000-\\u001f\\u007f-\\u009f\\u200b-\\u200f\\u2028-\\u202e\\u2066-\\u2069\\ufeff]',
  'g',
)

export const isHarpocratesReason = (reason: unknown): reason is string =>
  typeof reason === 'string' && (reason.startsWith(PREFIX) || reason.startsWith(LEGACY_PREFIX))

/** Unsafe characters become '?', so nothing drawn can move the cursor, recolour or reorder. */
export const sanitize = (text: string): string => text.replace(UNSAFE, '?')

/**
 * The reason's path (as written, not yet sanitised: it is compared with the
 * call's own path before anything is drawn) and kinds, or null when the reason
 * is not exactly the template the hook writes.
 */
export function parseReason(reason: string): ParsedAsk | null {
  const m = REASON.exec(reason.trimEnd())
  const path = m?.[1]
  const kinds = m?.[2]

  return path && kinds ? { path, kinds } : null
}

const isAbsolute = (path: string): boolean => path.startsWith('/') || /^[A-Za-z]:\//.test(path)

/** `path` with separators unified, joined to `cwd` when relative, "." and ".." resolved. */
function resolvePath(path: string, cwd: string | undefined): string | null {
  const unified = /^[A-Za-z]:[\\/]/.test(path) ? path.replace(/\\/g, '/') : path
  const full = isAbsolute(unified) || cwd === undefined ? unified : `${cwd.replace(/\\/g, '/')}/${unified}`
  const absolute = isAbsolute(full)
  const parts: string[] = []

  for (const part of full.split('/')) {
    if (part === '' || part === '.') continue
    if (part !== '..') parts.push(part)
    else if (parts.length > 0 && parts[parts.length - 1] !== '..') parts.pop()
    else if (!absolute) parts.push(part)
  }

  // A relative path with no cwd to anchor it, or a drive-less leading "..", cannot be compared.
  return absolute ? `/${parts.join('/')}` : cwd === undefined ? parts.join('/') : null
}

/**
 * Whether the reason's path and the call's path name one file. The hook prints
 * the path relative to the session's directory when inside it and absolute
 * otherwise; the call's may be either.
 */
export function isSameFile(reasonPath: string, inputPath: string, cwd: string | undefined): boolean {
  const a = resolvePath(reasonPath, cwd)
  const b = resolvePath(inputPath, cwd)
  return a !== null && a === b
}

const baseName = (path: string): string => truncateLeft(path.split(/[\\/]/).pop() || path, 60)

/** One line naming the file. */
export const questionFor = (ask: ParsedAsk): string =>
  `Allow Claude to edit ${baseName(ask.path)} without secret protection?`

/** `question`, or `question (2)`, `(3)`... for the first one `taken` does not hold. */
export function uniqueQuestion(question: string, taken: (q: string) => boolean): string {
  let candidate = question
  for (let n = 2; taken(candidate); n += 1) candidate = `${question} (${n})`
  return candidate
}

export const denyReason = (ask: ParsedAsk): string =>
  `The user kept ${ask.path} protected. Read it with mcp__harpo__safe_read; ask the user to make the edit.`

export const noAnswerReason = (ask: ParsedAsk): string =>
  `No one answered the Harpo prompt, so ${ask.path} stays protected. Read it with mcp__harpo__safe_read.`

export type CallLine = { readonly label: string; readonly path: string }

/** The target file of an edit call: its `file_path`, or a notebook's `notebook_path`. Nothing else of the input is read. */
export function targetPath(input: unknown): string | undefined {
  if (typeof input !== 'object' || input === null) return undefined
  const { file_path, notebook_path } = input as { file_path?: unknown; notebook_path?: unknown }
  const path = typeof file_path === 'string' ? file_path : notebook_path
  return typeof path === 'string' && path !== '' ? path : undefined
}

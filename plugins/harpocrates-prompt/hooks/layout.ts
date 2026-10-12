// The engine refuses a tree with more than MAX_ROWS rows around the dialog. It
// lays each text out at ROW_COLUMNS cells wide whatever the viewport's width
// (measured with `claude plugin test`: a card of up to 13 rows by this file's
// count draws, one of 14 may be refused), breaking at spaces and splitting a
// word longer than a row. This decides what the card shows so it stays inside
// the limit, giving up the least important text first: the hint, then the
// second row of the call line, then part of the path in the body. Widths are
// display cells (wide characters and emoji take two), see width.ts.

import type { CallLine, ParsedAsk } from './parse'
import { cellWidth, truncateLeft, truncateRight, widthOf } from './width'

export const MAX_ROWS = 13
export const ROW_COLUMNS = 40

export const TITLE = '🔒 Harpocrates'
export const SUBTITLE = ' — edit without secret protection'
export const SUBAGENT = ' · from a subagent'
export const NOTE =
  'Allowing lets Claude read and edit this file with its built-in tools for the rest of this session; these values go to the model provider as-is.'
export const HINT = 'Claude can still read it with safe_read without this.'

const BODY_BEFORE = 'Claude wants to edit '
const BODY_MIDDLE = ', which contains secrets ('
const BODY_AFTER = ').'
const BORDER_ROWS = 2
const BODY_FIXED = widthOf(BODY_BEFORE + BODY_MIDDLE + BODY_AFTER)
// Room kept for the path when a long list of kinds has to be cut.
const MIN_PATH = 16

/**
 * The rows `text` takes at ROW_COLUMNS cells wide: a greedy break at spaces,
 * and a word longer than a row starts a row of its own and is split by code
 * point. Never fewer than the engine's own count, so a card that fits here fits
 * there.
 */
export function rowsOf(text: string): number {
  let rows = 1
  let used = 0

  for (const word of text.split(' ')) {
    const width = widthOf(word)
    const need = used === 0 ? width : used + 1 + width

    if (need <= ROW_COLUMNS) {
      used = need
    } else if (width <= ROW_COLUMNS) {
      rows += 1
      used = width
    } else {
      if (used > 0) rows += 1
      used = 0
      for (const ch of word) {
        const w = cellWidth(ch.codePointAt(0) ?? 0)
        if (used + w > ROW_COLUMNS) {
          rows += 1
          used = 0
        }
        used += w
      }
    }
  }

  return rows
}

export type Card = {
  readonly header: string
  /** The call line: the dim label part, then the target. */
  readonly call: { readonly label: string; readonly rest: string }
  readonly body: string
  readonly note: string
  readonly hasHint: boolean
  /** The rows the engine will count for the card around the dialog. */
  readonly rows: number
}

/** `render(path)` for the longest suffix of `path` (cut at the left) that keeps it within `maxRows`. */
function fitPath(maxRows: number, path: string, render: (path: string) => string): string {
  const fits = (cells: number): boolean => rowsOf(render(truncateLeft(path, cells))) <= maxRows
  let low = 1
  let high = widthOf(path)

  // The largest width that fits, found by halving; wrapping at spaces can make
  // it uneven, so step down from there until it does fit.
  while (low < high) {
    const mid = Math.ceil((low + high) / 2)
    if (fits(mid)) low = mid
    else high = mid - 1
  }

  for (let cells = low; cells > 1; cells -= 1) {
    if (fits(cells)) return render(truncateLeft(path, cells))
  }

  return render('…')
}

/** What the card shows, fitted to the engine's row limit (see the top of this file). */
export function buildCard(args: {
  ask: ParsedAsk
  call: CallLine
  isSubagent: boolean
  columns: number
}): Card {
  const { ask, call, isSubagent, columns } = args
  const inner = Math.max(20, columns - 4)
  const header = TITLE + SUBTITLE + (isSubagent ? SUBAGENT : '')
  const labelW = widthOf(call.label) + 2
  const fixedRows = BORDER_ROWS + rowsOf(header) + rowsOf(NOTE)
  // Most wanted first: the hint goes before the call line shrinks, which goes
  // before the body does.
  const options = [
    { bodyRows: 3, callRows: 2, hasHint: true },
    { bodyRows: 3, callRows: 2, hasHint: false },
    { bodyRows: 3, callRows: 1, hasHint: false },
    { bodyRows: 2, callRows: 1, hasHint: false },
  ]
  let last: Card | undefined

  for (const { bodyRows, callRows, hasHint } of options) {
    const label = call.label + ' '.repeat(2)
    const limit = Math.min(inner, callRows * ROW_COLUMNS)
    const kinds = truncateRight(ask.kinds, Math.max(1, bodyRows * ROW_COLUMNS - BODY_FIXED - MIN_PATH))
    const target = fitPath(callRows, call.path, p => label + truncateLeft(p, Math.max(1, limit - labelW)))
    const body = fitPath(bodyRows, ask.path, p => `${BODY_BEFORE}${p}${BODY_MIDDLE}${kinds}${BODY_AFTER}`)
    const rows = fixedRows + rowsOf(target) + rowsOf(body) + (hasHint ? rowsOf(HINT) : 0)

    last = { header, call: { label, rest: target.slice(label.length) }, body, note: NOTE, hasHint, rows }
    if (rows <= MAX_ROWS) return last
  }

  return last as Card
}

// Display width in terminal cells, with no dependency. East Asian wide and
// fullwidth characters and emoji take two cells. The table errs on the wide
// side (some symbols that many terminals draw narrow count two): a card that
// fits by this count fits when drawn. Text is walked by code point, so a
// surrogate pair is never split.

const WIDE: readonly (readonly [number, number])[] = [
  [0x1100, 0x115f], // Hangul Jamo
  [0x231a, 0x231b],
  [0x2329, 0x232a],
  [0x23e9, 0x23f3],
  [0x25fd, 0x25fe],
  [0x2600, 0x27bf], // symbols and dingbats, many drawn as emoji
  [0x2b1b, 0x2b1c],
  [0x2b50, 0x2b55],
  [0x2e80, 0x303e], // CJK radicals, punctuation
  [0x3041, 0x33ff], // kana, CJK symbols
  [0x3400, 0x4dbf], // CJK extension A
  [0x4e00, 0xa4cf], // CJK unified ideographs, Yi
  [0xa960, 0xa97f],
  [0xac00, 0xd7a3], // Hangul syllables
  [0xf900, 0xfaff], // CJK compatibility ideographs
  [0xfe10, 0xfe19],
  [0xfe30, 0xfe6f],
  [0xff00, 0xff60], // fullwidth forms
  [0xffe0, 0xffe6],
  [0x1b000, 0x1b2ff],
  [0x1f000, 0x1faff], // emoji and pictographs
  [0x20000, 0x3fffd], // CJK extensions B and up
]

export function cellWidth(codePoint: number): 1 | 2 {
  for (const [from, to] of WIDE) {
    if (codePoint >= from && codePoint <= to) return 2
    if (codePoint < from) break
  }

  return 1
}

const ELLIPSIS = '…'

/** The cells `text` takes. */
export function widthOf(text: string): number {
  let width = 0
  for (const ch of text) width += cellWidth(ch.codePointAt(0) ?? 0)
  return width
}

/** `text` cut at the left with a leading ellipsis, within `max` cells. */
export function truncateLeft(text: string, max: number): string {
  if (widthOf(text) <= max) return text
  if (max <= 1) return ELLIPSIS

  const kept: string[] = []
  let width = 1
  for (const ch of [...text].reverse()) {
    width += cellWidth(ch.codePointAt(0) ?? 0)
    if (width > max) break
    kept.push(ch)
  }

  return ELLIPSIS + kept.reverse().join('')
}

/** `text` cut at the right with a trailing ellipsis, within `max` cells. */
export function truncateRight(text: string, max: number): string {
  if (widthOf(text) <= max) return text
  if (max <= 1) return ELLIPSIS

  const kept: string[] = []
  let width = 1
  for (const ch of text) {
    width += cellWidth(ch.codePointAt(0) ?? 0)
    if (width > max) break
    kept.push(ch)
  }

  return kept.join('') + ELLIPSIS
}

import { expect, test, tier } from 'claude-code/testing'
import type { Engine } from 'claude-code/testing'
import type { On, ToolCheckResult } from 'claude-code'

import { buildCard, MAX_ROWS } from '../hooks/layout'
import { isSameFile, parseReason, sanitize } from '../hooks/parse'
import { truncateLeft, truncateRight, widthOf } from '../hooks/width'

tier('user')

const KEEP = 'Keep secrets protected'
const ALLOW = 'Allow edits this session'

const reasonFor = (path: string, kinds = 'api_token, password') =>
  `Harpocrates: Claude wants to edit ${path} without secret protection. It contains secrets (${kinds}). ` +
  'Allowing lets Claude read and edit this file with its built-in tools for the rest of this session; ' +
  'these values go to the model provider as-is. Choose No to keep them protected: ' +
  'Claude can still read it with mcp__harpocrates__safe_read.'

const REASON = reasonFor('config/settings.py')

type Asked = { question: string; header: string; options: string[] }
const DISMISS = Symbol('dismiss')
type Answer = string | typeof DISMISS

const ask = (reason: string): ToolCheckResult => ({ decision: 'ask', reason, hook: 'PreToolUse' })

// The engine beneath the plugin: the verdict it reaches, the tool it runs, and
// what the person answers when the AskUserQuestion dialog opens. `draw` may
// mount the dialog while it is open and read what the plugin drew.
function world(
  on: On,
  verdict: ToolCheckResult | ((tool: string) => ToolCheckResult),
  answer: Answer | ((question: string) => Answer),
  draw?: (question: string) => Promise<void>,
) {
  const asked: Asked[] = []
  on('tool.check', (_$, e) => (typeof verdict === 'function' ? verdict(e.tool) : verdict))
  on('tool.call', { tool: 'AskUserQuestion' }, async (_$, e) => {
    const q = e.questions[0]
    if (!q) return { deny: 'no question' }
    asked.push({ question: q.question, header: q.header, options: q.options.map(o => o.label) })
    await draw?.(q.question)
    const a = typeof answer === 'function' ? answer(q.question) : answer
    if (a === DISMISS) return { deny: 'dismissed' }
    return { result: { questions: e.questions, answers: { [q.question]: a } } }
  })
  return { asked }
}

type Verdict = { decision: string; reason?: string }
type Checker = { tool: { check: (a: never) => Promise<Verdict> } }
let calls = 0

// A real call as the engine raises it: the check carries the call's id.
const run = ($: unknown, tool: string, input: object, extra: object = {}) =>
  ($ as Checker).tool.check({ tool, input, tool_use_id: `tu${(calls += 1)}`, ...extra } as never)
const edit = ($: unknown, file_path = 'config/settings.py', extra: object = {}) =>
  run($, 'Edit', { file_path, old_string: 'a', new_string: 'b' }, extra)

// -- parse --------------------------------------------------------------

test('parse: a plain reason gives the path and the kinds', () => {
  expect(parseReason(REASON)).toEqual({ path: 'config/settings.py', kinds: 'api_token, password' })
})

test('parse: spaces and parentheses in the path are part of the path', () => {
  expect(parseReason(reasonFor('/Users/me/My Project (old)/.env (copy).local', 'aws_access_key'))).toEqual({
    path: '/Users/me/My Project (old)/.env (copy).local',
    kinds: 'aws_access_key',
  })
})

test('parse: a path that spells the anchor still splits at the real kinds', () => {
  const path = 'a without secret protection. It contains secrets (password). Allowing b'
  expect(parseReason(reasonFor(path, 'private_key'))).toEqual({ path, kinds: 'private_key' })
})

test('parse: the unscanned kind', () => {
  expect(parseReason(reasonFor('big.log', 'unscanned (over the size limit)'))).toEqual({
    path: 'big.log',
    kinds: 'unscanned (over the size limit)',
  })
})

test('parse: CRLF line ends are trimmed, and the path is returned as written', () => {
  expect(parseReason(REASON + '\r\n')).toEqual({ path: 'config/settings.py', kinds: 'api_token, password' })
  expect(parseReason(reasonFor('a\u001b[31mb.env'))?.path).toBe('a\u001b[31mb.env')
})

test('sanitize: control, bidi, zero-width and separator characters become ?', () => {
  expect(sanitize('a\u001b[31mb')).toBe('a?[31mb')
  // A right-to-left override makes "evil\u202Efdp.exe" display as "evilexe.pdf".
  expect(sanitize('evil\u202Efdp.exe')).toBe('evil?fdp.exe')
  expect(sanitize('a\u2066b\u2069c\u202ad\u202ce')).toBe('a?b?c?d?e')
  expect(sanitize('a\u200bb\u200fc\ufeffd\u2028e\u2029f')).toBe('a?b?c?d?e?f')
  expect(sanitize('plain/path (1)/文件.env 🔒')).toBe('plain/path (1)/文件.env 🔒')
})

test('isSameFile: relative and absolute forms of one file, resolved against the cwd', () => {
  expect(isSameFile('src/a.env', 'src/a.env', undefined)).toBe(true)
  expect(isSameFile('src/a.env', './src/a.env', undefined)).toBe(true)
  expect(isSameFile('src/a.env', '/proj/src/a.env', '/proj')).toBe(true)
  expect(isSameFile('src/a.env', '/proj/src/../src/a.env', '/proj')).toBe(true)
  expect(isSameFile('/etc/a.env', '/etc/a.env', '/proj')).toBe(true)
  expect(isSameFile('/etc/a.env', '/etc//./a.env', undefined)).toBe(true)
  expect(isSameFile('a b/(c).env', '/proj/a b/(c).env', '/proj')).toBe(true)
  expect(isSameFile('C:/p/a.env', 'C:\\p\\a.env', undefined)).toBe(true)
})

test('isSameFile: different files, and relative against absolute with no cwd, are not the same', () => {
  expect(isSameFile('src/a.env', 'src/b.env', undefined)).toBe(false)
  expect(isSameFile('src/a.env', '/proj/src/b.env', '/proj')).toBe(false)
  expect(isSameFile('src/a.env', '/other/src/a.env', '/proj')).toBe(false)
  expect(isSameFile('src/a.env', '/proj/src/a.env', undefined)).toBe(false)
  expect(isSameFile('src/a.env', '../src/a.env', '/proj')).toBe(false)
  expect(isSameFile('a.env', 'a.env/', undefined)).toBe(true)
})

test('parse: anything else is null', () => {
  for (const garbage of [
    '',
    'Harpocrates: something new we cannot read',
    'Harpocrates: Claude wants to read files that contain secrets:\n  a.env (password)\nAllowing sends these values',
    reasonFor('x.env', 'Not A Kind'),
    reasonFor('x.env', ''),
    reasonFor('', 'password'),
    REASON.replace('Allowing lets', 'Permitting lets'),
    REASON.replace('Harpocrates: ', ''),
  ]) {
    expect(parseReason(garbage)).toBeNull()
  }
})

// -- decisions ----------------------------------------------------------

test('a Harpocrates ask opens our question with the two options', async ($, on) => {
  const w = world(on, ask(REASON), KEEP)
  await edit($)
  expect(w.asked).toEqual([
    {
      question: 'Allow Claude to edit settings.py without secret protection?',
      header: 'Harpocrates',
      options: [KEEP, ALLOW],
    },
  ])
})

test('Allow edits this session allows', async ($, on) => {
  world(on, ask(REASON), ALLOW)
  expect((await edit($)).decision).toBe('allow')
})

test('Keep secrets protected denies and names the path and safe_read', async ($, on) => {
  world(on, ask(REASON), KEEP)
  const r = await edit($)
  expect(r.decision).toBe('deny')
  expect(r.reason).toBe(
    'The user kept config/settings.py protected. Read it with mcp__harpocrates__safe_read; ask the user to make the edit.',
  )
})

test('free text under Other, and a near-miss label, deny', async ($, on) => {
  let said = ''
  const w = world(on, ask(REASON), () => said)
  for (const answer of ['allow edits this session', 'Allow edits this session please', `${ALLOW}, ${KEEP}`]) {
    said = answer
    expect((await edit($)).decision).toBe('deny')
  }
  expect(w.asked.length).toBe(3)
})

test('a dismissed question denies with the no-answer reason and logs only the tool', async ($, on) => {
  const logs: string[] = []
  on('ui.log', (_$, e) => {
    logs.push(`${e.to}: ${e.text}`)
    return { value: undefined } as never
  })
  world(on, ask(REASON), DISMISS)
  const r = await edit($)
  expect(logs).toEqual(['debug: harpocrates-prompt: question not answered (Edit)'])
  expect(r.decision).toBe('deny')
  expect(r.reason).toBe(
    'No one answered the Harpocrates prompt, so config/settings.py stays protected. Read it with mcp__harpocrates__safe_read.',
  )
})

test('two pending asks for one file get distinct questions', async ($, on) => {
  const w = world(on, ask(REASON), KEEP)
  await Promise.all([edit($), edit($)])
  const questions = w.asked.map(a => a.question)
  expect(new Set(questions).size).toBe(2)
})

// -- left to the engine -------------------------------------------------

test('verdicts that are not a Harpocrates ask pass through untouched', async ($, on) => {
  const cases: ToolCheckResult[] = [
    { decision: 'ask', reason: 'Bash needs approval', hook: 'PreToolUse' },
    { decision: 'ask', reason: REASON },
    { decision: 'ask', reason: REASON, hook: 'PostToolUse' },
    { decision: 'allow', reason: REASON, hook: 'PreToolUse' },
    { decision: 'deny', reason: 'Harpocrates: blocked', hook: 'PreToolUse' },
  ]
  let current = cases[0] as ToolCheckResult
  const w = world(on, () => current, ALLOW)
  for (const verdict of cases) {
    current = verdict
    const v = await edit($)
    expect(v.decision).toBe(verdict.decision)
    expect(v.reason).toBe(verdict.reason)
  }
  expect(w.asked.length).toBe(0)
})

test('an unparseable Harpocrates reason returns the original verdict: no question', async ($, on) => {
  const odd = [
    'Harpocrates: something new we cannot read',
    'Harpocrates: Claude wants to read files that contain secrets:\n  a.env (password)\nAllowing sends these values',
    reasonFor('x.env', 'Not A Kind'),
  ]
  let current = odd[0] as string
  const w = world(on, () => ask(current), ALLOW)
  for (const reason of odd) {
    current = reason
    expect(await edit($)).toMatchObject({ decision: 'ask', reason, hook: 'PreToolUse' })
  }
  expect(w.asked.length).toBe(0)
})

test('a query (no tool_use_id) never opens a dialog', async ($, on) => {
  const w = world(on, ask(REASON), ALLOW)
  const v = await $.tool.check({ tool: 'Edit', input: { file_path: '/x' } })
  expect(v.decision).toBe('ask')
  expect(w.asked.length).toBe(0)
})

test('an error beneath the hook never turns into allow', async ($, on) => {
  on('tool.check', () => {
    throw new Error('boom')
  })
  const outcome = await $.tool.check({ tool: 'Edit', input: {} }).then(
    v => v.decision,
    () => 'rejected',
  )
  expect(outcome).not.toBe('allow')
})

// -- the card must describe the real call --------------------------------

test('a Harpocrates edit reason on a tool that is not an edit is left to the engine', async ($, on) => {
  const w = world(on, ask(REASON), ALLOW)
  for (const [tool, input] of [
    ['Bash', { command: 'echo hi > config/settings.py' }],
    ['Read', { file_path: 'config/settings.py' }],
    ['Grep', { pattern: 'x', path: 'config/settings.py' }],
  ] as const) {
    expect(await run($, tool, input)).toMatchObject({ decision: 'ask', reason: REASON, hook: 'PreToolUse' })
  }
  expect(w.asked.length).toBe(0)
})

test('an edit with no path in its input is left to the engine', async ($, on) => {
  const w = world(on, ask(REASON), ALLOW)
  for (const input of [{}, { file_path: '' }, { file_path: 3 }, { old_string: 'a' }]) {
    expect(await run($, 'Edit', input)).toMatchObject({ decision: 'ask', reason: REASON })
  }
  expect(w.asked.length).toBe(0)
})

test('an edit whose path is not the one in the reason is left to the engine', async ($, on) => {
  const w = world(on, ask(REASON), ALLOW)
  for (const file_path of ['config/other.py', '/config/settings.py', 'x/config/settings.py', '../config/settings.py']) {
    expect(await edit($, file_path)).toMatchObject({ decision: 'ask', reason: REASON })
  }
  expect(w.asked.length).toBe(0)
})

test('the same file in another spelling is the same file', async ($, on) => {
  const w = world(on, ask(reasonFor('src/a.env', 'password')), KEEP)
  for (const file_path of ['src/a.env', './src/a.env', 'src//a.env', 'src/./a.env']) {
    expect((await edit($, file_path)).decision).toBe('deny')
  }
  expect(w.asked.length).toBe(4)
})

test('every edit tool is drawn for, and a notebook through notebook_path', async ($, on) => {
  const w = world(on, ask(reasonFor('n/a.ipynb')), KEEP)
  for (const tool of ['Edit', 'Write', 'MultiEdit']) {
    expect((await run($, tool, { file_path: 'n/a.ipynb' })).decision).toBe('deny')
  }
  expect((await run($, 'NotebookEdit', { notebook_path: 'n/a.ipynb' })).decision).toBe('deny')
  expect(w.asked.length).toBe(4)
})

test('the question and the deny text use the call\'s own path', async ($, on) => {
  const w = world(on, ask(reasonFor('a b/(c).env')), KEEP)
  const r = await edit($, './a b/(c).env')
  expect(w.asked[0]?.question).toBe('Allow Claude to edit (c).env without secret protection?')
  expect(r.reason).toBe(
    'The user kept ./a b/(c).env protected. Read it with mcp__harpocrates__safe_read; ask the user to make the edit.',
  )
})

test('a verdict that names a settings rule is left to the engine', async ($, on) => {
  const w = world(on, { decision: 'ask', reason: REASON, hook: 'PreToolUse', rule: 'Edit(*)' }, ALLOW)
  expect(await edit($)).toMatchObject({ decision: 'ask', rule: 'Edit(*)' })
  expect(w.asked.length).toBe(0)
})

test('a reason with any change to the template is left to the engine', async ($, on) => {
  const changed = [
    REASON.replace('Allowing lets Claude read and edit this file', 'Allowing lets Claude read this file'),
    REASON.replace('with its built-in tools', 'with any tools'),
    REASON.replace('Choose No to keep them protected', 'Choose No'),
    REASON + ' Extra.',
    REASON.replace('mcp__harpocrates__safe_read', 'mcp__harpocrates__safe_grep'),
    REASON.replace('(api_token, password)', '(API_TOKEN)'),
    REASON.replace('(api_token, password)', '(api_token,password)'),
  ]
  let current = changed[0] as string
  const w = world(on, () => ask(current), ALLOW)
  for (const reason of changed) {
    current = reason
    expect(await edit($)).toMatchObject({ decision: 'ask', reason })
  }
  expect(w.asked.length).toBe(0)
})

test('a disguised extension is drawn with the disguise removed', async ($, on) => {
  on('ui.render', () => ({ type: 'engine', ref: 1 }))
  const name = 'invoice\u202Etxt.exe'
  let drawn: boolean[] = []
  const w = world(on, ask(reasonFor(`/d/${name}`)), KEEP, async question => {
    const { ui, has } = await card($, 'terminal', question)
    drawn = [await has(/\u202e/), await has(/invoice\?txt\.exe/)]
    await ui.unmount()
  })
  await edit($, `/d/${name}`)
  expect(drawn).toEqual([false, true])
  expect(w.asked[0]?.question).not.toMatch(/\u202e/)
})

// -- the card -----------------------------------------------------------

const dialogProps = (question: string, header = 'Harpocrates') => ({
  tool: 'AskUserQuestion',
  questions: [
    {
      question,
      header,
      options: [{ label: KEEP }, { label: ALLOW, description: 'kept' }],
      multiSelect: false,
    },
  ],
})

// Draws the dialog for `question` while it is pending and returns every text.
async function card($: Engine, surface: 'terminal' | 'desktop', question: string, columns = 100) {
  const ui = await $.ui.mount({
    plugin: 'harpocrates-prompt',
    surface,
    component: 'AskUserQuestion',
    viewport: { columns, rows: 40 },
    props: dialogProps(question),
  } as never)
  const has = async (text: RegExp) => (await ui.find({ type: 'Text', text })) !== undefined
  return { ui, has }
}

for (const surface of ['terminal', 'desktop'] as const) {
  test(`${surface}: the dialog is wrapped in our card while the question is pending`, async ($, on) => {
    on('ui.render', () => ({ type: 'engine', ref: 1 }))
    let isDrawn = false
    world(on, ask(reasonFor('a.env', 'password')), ALLOW, async question => {
      const { ui, has } = await card($, surface, question)
      for (const text of [
        /Harpocrates/,
        /edit without secret protection/,
        /Edit\s+a\.env/,
        /Claude wants to edit a\.env, which contains secrets \(password\)\./,
        /Allowing lets Claude read and edit this file with its built-in tools for the rest of this session; these values go to the model provider as-is\./,
        /Claude can still read it with safe_read without this\./,
      ]) {
        expect(await has(text)).toBe(true)
      }
      expect(await has(/from a subagent/)).toBe(false)
      await ui.unmount()
      isDrawn = true
    })
    expect((await edit($, 'a.env')).decision).toBe('allow')
    expect(isDrawn).toBe(true)
  })
}

test('a subagent call is marked in the header', async ($, on) => {
  on('ui.render', () => ({ type: 'engine', ref: 1 }))
  let isMarked = false
  world(on, ask(REASON), KEEP, async question => {
    const { ui, has } = await card($, 'terminal', question)
    isMarked = await has(/from a subagent/)
    await ui.unmount()
  })
  await edit($)
  expect(isMarked).toBe(false)
  await edit($, undefined, { agentId: 'sub-1' })
  expect(isMarked).toBe(true)
})

test('the option descriptions are added to the engine dialog', async ($, on) => {
  let seen: unknown
  on('ui.render', { component: 'AskUserQuestion' }, (_$, e) => {
    seen = (e.props.questions[0] as { options?: unknown } | undefined)?.options
    return { type: 'engine', ref: 1 } as never
  })
  world(on, ask(REASON), KEEP, async question => {
    const { ui } = await card($, 'terminal', question)
    await ui.unmount()
  })
  await edit($)
  expect(seen).toEqual([
    { label: KEEP, description: 'Claude keeps using safe_read; the file stays unedited' },
    { label: ALLOW, description: 'Built-in Read and Edit of this file, until the session ends' },
  ])
})

test('a dialog that is not ours is drawn by the engine untouched', async ($, on) => {
  on('ui.render', () => ({ type: 'engine', ref: 1 }))
  const ui = await $.ui.mount({
    plugin: 'harpocrates-prompt',
    surface: 'terminal',
    component: 'AskUserQuestion',
    viewport: { columns: 100, rows: 40 },
    props: dialogProps('Allow Claude to edit foo without secret protection?'),
  } as never)
  expect((await ui.find({ type: 'Text', text: /Harpocrates/ })) !== undefined).toBe(false)
  await ui.unmount()
})

test('the card shows the target path and none of the rest of the tool input', async ($, on) => {
  on('ui.render', () => ({ type: 'engine', ref: 1 }))
  const SENTINEL = 'ZZ-SENTINEL-CONTENT'
  let drawn: boolean[] = []
  world(on, ask(reasonFor('/work/config.py')), KEEP, async question => {
    const { ui, has } = await card($, 'terminal', question)
    drawn = [await has(/SENTINEL|old text|new text|file body/), await has(/\/work\/config\.py/)]
    await ui.unmount()
  })
  await run($, 'Write', {
    file_path: '/work/config.py',
    content: `file body ${SENTINEL}`,
    old_string: `old text ${SENTINEL}`,
    new_string: `new text ${SENTINEL}`,
    command: SENTINEL,
  })
  expect(drawn).toEqual([false, true])
})

test('NotebookEdit shows its notebook path', async ($, on) => {
  on('ui.render', () => ({ type: 'engine', ref: 1 }))
  let isShown = false
  world(on, ask(reasonFor('/n/a.ipynb')), KEEP, async question => {
    const { ui, has } = await card($, 'terminal', question)
    isShown = await has(/NotebookEdit\s+\/n\/a\.ipynb/)
    await ui.unmount()
  })
  await run($, 'NotebookEdit', { notebook_path: '/n/a.ipynb', new_source: 'secret = 1' })
  expect(isShown).toBe(true)
})

test('control characters in the target never reach the card', async ($, on) => {
  on('ui.render', () => ({ type: 'engine', ref: 1 }))
  let drawn: boolean[] = []
  world(on, ask(reasonFor('/x\u001b[31m\u202Efdp.exe')), KEEP, async question => {
    const { ui, has } = await card($, 'terminal', question)
    drawn = [await has(/[\u001b\u202e]/), await has(/Edit\s+\/x\?\[31m\?fdp\.exe/)]
    await ui.unmount()
  })
  await edit($, '/x\u001b[31m\u202Efdp.exe')
  expect(drawn).toEqual([false, true])
})

// -- display width ------------------------------------------------------

test('width: wide characters and emoji are two cells, the rest one', () => {
  expect(widthOf('abc')).toBe(3)
  expect(widthOf('文件')).toBe(4)
  expect(widthOf('한글')).toBe(4)
  expect(widthOf('Ｆ')).toBe(2)
  expect(widthOf('🔒')).toBe(2)
  expect(widthOf('a🔒b')).toBe(4)
})

test('width: truncation counts cells and never splits a surrogate pair', () => {
  const text = '🔒🔑🔒🔑🔒🔑'
  const lone = /[\ud800-\udbff](?![\udc00-\udfff])|(?<![\ud800-\udbff])[\udc00-\udfff]/
  for (let max = 1; max <= 14; max += 1) {
    for (const cut of [truncateLeft(text, max), truncateRight(text, max)]) {
      expect(widthOf(cut)).toBeLessThanOrEqual(max)
      expect(lone.test(cut)).toBe(false)
    }
  }
  expect(truncateLeft('/a/文件夹/密钥.env', 10)).toBe('…/密钥.env')
  expect(truncateRight('文件夹文件夹', 7)).toBe('文件夹…')
  expect(truncateLeft('short', 10)).toBe('short')
})

// -- row-cap fitting ----------------------------------------------------

test('layout: every card fits the row cap, whatever the path, kinds, viewport or subagent', () => {
  const refused: string[] = []
  for (const [segment, tail] of [['/d/', 'file.env'], ['/文件夹/', '密钥.env'], ['/🔒🔑/', '🔑.env'], ['/a b/', 'x y.env']] as const) {
   for (const depth of [0, 1, 4, 12, 40]) {
    const path = segment.repeat(depth) + tail
    for (const kinds of ['password', 'api_token, password', 'unscanned (over the size limit)', Array(30).fill('aws_access_key').join(', ')]) {
      for (const columns of [24, 50, 80, 200]) {
        for (const isSubagent of [false, true]) {
          const c = buildCard({
            ask: { path, kinds },
            call: { label: 'NotebookEdit', path },
            isSubagent,
            columns,
          })
          if (c.rows > MAX_ROWS) refused.push(`depth=${depth} cols=${columns} sub=${isSubagent} rows=${c.rows}`)
        }
      }
    }
   }
  }
  expect(refused).toEqual([])
})

test('layout: the hint is the first thing to go, the full path survives while it fits', () => {
  const short = buildCard({
    ask: { path: 'a.env', kinds: 'password' },
    call: { label: 'Edit', path: 'a.env' },
    isSubagent: false,
    columns: 80,
  })
  expect(short.hasHint).toBe(true)
  const long = buildCard({
    ask: { path: '/my project/config/settings.py', kinds: 'api_token, password' },
    call: { label: 'Edit', path: '/my project/config/settings.py' },
    isSubagent: true,
    columns: 80,
  })
  expect(long.hasHint).toBe(false)
  expect(long.body).toContain('/my project/config/settings.py')
})

test('the card is never refused for size, whatever the path, kinds or width', async ($, on) => {
  on('ui.render', () => ({ type: 'engine', ref: 1 }))
  let reason = ''
  let isDrawn = false
  let columns = 80
  on('tool.check', () => ask(reason))
  on('tool.call', { tool: 'AskUserQuestion' }, async (_$, e) => {
    const q = e.questions[0]
    if (!q) return { deny: 'no question' }
    const ui = await $.ui.mount({
      plugin: 'harpocrates-prompt',
      surface: 'terminal',
      component: 'AskUserQuestion',
      viewport: { columns, rows: 40 },
      props: dialogProps(q.question),
    } as never)
    isDrawn = (await ui.find({ type: 'Text', text: /edit without secret protection/ })) !== undefined
    await ui.unmount()
    return { result: { questions: e.questions, answers: { [q.question]: KEEP } } }
  })
  const refused: string[] = []
  for (const segment of ['/dir (x)/', '/文件夹/', '/🔒🔑/']) {
   for (const depth of [0, 3, 15, 50]) {
    for (const kinds of ['password', 'unscanned (over the size limit)', Array(20).fill('api_token').join(', ')]) {
      for (const width of [30, 50, 80, 200]) {
        for (const agentId of [undefined, 'sub']) {
          const path = segment.repeat(depth) + 'f.env'
          reason = reasonFor(path, kinds)
          columns = width
          isDrawn = false
          await run($, 'Edit', { file_path: path }, agentId ? { agentId } : {})
          if (!isDrawn) refused.push(`depth=${depth} width=${width} sub=${!!agentId} kinds=${kinds.length}`)
        }
      }
    }
  }
   }
  expect(refused).toEqual([])
})

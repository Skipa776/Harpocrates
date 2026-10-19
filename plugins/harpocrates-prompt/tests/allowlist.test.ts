import { expect, test, tier } from 'claude-code/testing'
import type { Engine } from 'claude-code/testing'
import type { On, PromptOrigin } from 'claude-code'

import {
  HELP,
  absolutePath,
  changeArgv,
  listText,
  parseAllowArgs,
  tokenize,
} from '../hooks/allowlist'

tier('user')

const NOT_FOUND = 'Harpocrates CLI not found. Install it: pip install git+https://github.com/Skipa776/harpocrates'
const REFUSED = 'Only you can change the Harpocrates allowlist: type /harpo in the chat box.'
const EVERYWHERE = 'Allow it everywhere'
const PRESENTATION = { isFullscreen: false, columns: 80 }
const PY = '/venv/bin/python'
const CLI = [PY, '-P', '-m', 'Harpocrates.cli']
const RECORD = JSON.stringify({ version: 1, python: PY, python_args: ['-P'], 'claude-code': { mcp: true } })
const NOT_INSTALLED =
  'Run `harpo install` first: the plugin only runs the Harpocrates interpreter that install recorded.'

type Reply = { exitCode: number; stdout: string } | 'missing'

const json = (value: unknown, exitCode = 0): Reply => ({ exitCode, stdout: JSON.stringify(value) })

// The engine beneath the plugin: the CLI (every call recorded, answered by
// `reply`), the session folder and home, and the answer the person gives the
// dialog (recorded in `asked`).
function world(on: On, first: (argv: readonly string[]) => Reply, firstAnswer: string | null = null) {
  const ran: (readonly string[])[] = []
  const read: string[] = []
  const asked: { question: string; options: string[] }[] = []
  let reply = first
  let answer = firstAnswer
  let record: string | null = RECORD
  let recordAt: Record<string, string> | null = null

  // The install record: a missing file rejects the read, as the engine's does.
  on('fs.read', (_$, e) => {
    read.push(e.path)
    const text = recordAt === null ? record : (recordAt[e.path] ?? null)
    return text === null ? { deny: 'ENOENT' } : { value: text }
  })
  on('process.run', (_$, e) => {
    ran.push(e.argv)
    const r = reply(e.argv)
    // A denied process.run rejects the caller, as a command that cannot start does.
    if (r === 'missing') return { deny: 'spawn ENOENT' }
    return {
      value: { exitCode: r.exitCode, stdout: r.stdout, stderr: '', isStdoutTruncated: false, isStderrTruncated: false },
    }
  })
  on('session.cwd', () => ({ value: '/proj/app' }))
  on('env.get', (_$, e) => ({ value: e.name === 'HOME' ? '/home/me' : undefined }))
  on('tool.call', { tool: 'AskUserQuestion' }, (_$, e) => {
    const q = e.questions[0]
    if (!q) return { deny: 'no question' }
    asked.push({ question: q.question, options: q.options.map(o => o.label) })
    if (answer === null) return { deny: 'dismissed' }
    return { result: { questions: e.questions, answers: { [q.question]: answer } } }
  })

  return {
    ran,
    asked,
    read,
    /** What the install record holds (null: no file) from now on. */
    setRecord(text: string | null) {
      record = text
      recordAt = null
    },
    /** What each folder's install record holds, by path (a path left out has no file) from now on. */
    setRecordAt(files: Record<string, string>) {
      recordAt = files
    },
    /** Changes what the CLI answers (and, optionally, what the person picks) from now on; clears the record. */
    use(next: (argv: readonly string[]) => Reply, nextAnswer: string | null = null) {
      reply = next
      answer = nextAnswer
      ran.length = 0
      asked.length = 0
    },
  }
}

// The slash command has two names (FR-NAME-02): /harpo, and /harpocrates for two more minor releases.
const NAMES = ['harpo', 'harpocrates'] as const

const say = async ($: Engine, args: string, origin: PromptOrigin = { kind: 'composer' }, command: string = NAMES[0]) =>
  (await $.command.run({ command, args, origin, presentation: PRESENTATION })).text

const ADDED = json({ ok: true, action: 'added', kind: 'path', value: '/proj/app/.env' })

// -- parsing ----------------------------------------------------------------

test('tokenize: quotes group words and are dropped; an unclosed quote is null', () => {
  expect(tokenize('allow "my dir/a b.env"')).toEqual(['allow', 'my dir/a b.env'])
  expect(tokenize("allow 'it is' x")).toEqual(['allow', 'it is', 'x'])
  expect(tokenize('a"b c"d e')).toEqual(['ab cd', 'e'])
  expect(tokenize('a "" b')).toEqual(['a', '', 'b'])
  expect(tokenize('  C:\\Users\\me  ')).toEqual(['C:\\Users\\me'])
  expect(tokenize('')).toEqual([])
  expect(tokenize('allow "oops')).toBeNull()
})

test('parse: each subcommand and its aliases', () => {
  expect(parseAllowArgs('')).toEqual({ kind: 'help' })
  expect(parseAllowArgs('help')).toEqual({ kind: 'help' })
  expect(parseAllowArgs('allowlist')).toEqual({ kind: 'list' })
  expect(parseAllowArgs('list')).toEqual({ kind: 'list' })
  expect(parseAllowArgs('allow --list')).toEqual({ kind: 'list' })
  expect(parseAllowArgs('allow a/b')).toEqual({ kind: 'add', target: { kind: 'path', value: 'a/b' } })
  expect(parseAllowArgs('allow --name graphify-out')).toEqual({
    kind: 'add',
    target: { kind: 'name', value: 'graphify-out' },
  })
  expect(parseAllowArgs('remove "a b"')).toEqual({ kind: 'remove', target: { kind: 'path', value: 'a b' } })
  expect(parseAllowArgs('remove --name x')).toEqual({ kind: 'remove', target: { kind: 'name', value: 'x' } })
  expect(parseAllowArgs('ALLOW a')).toEqual({ kind: 'add', target: { kind: 'path', value: 'a' } })
})

test('parse: unknown words, bad shapes and flags the command does not offer', () => {
  expect(parseAllowArgs('frobnicate x')).toEqual({ kind: 'unknown', word: 'frobnicate' })
  for (const bad of [
    'allow',
    'remove',
    'allow --name',
    'allow --name a b',
    'allow --name -x',
    'allow --yes',
    'allow --yes a',
    'allow a b',
    'allow ""',
    'allow "oops',
    'allow a\u0000b',
    'remove --list',
  ]) {
    expect(parseAllowArgs(bad).kind).toBe('usage')
  }
})

test('absolutePath: home, absolute and relative, with no base to anchor them', () => {
  expect(absolutePath('~/x', '/proj', '/home/me/')).toBe('/home/me/x')
  expect(absolutePath('~', '/proj', '/home/me')).toBe('/home/me')
  expect(absolutePath('~/x', '/proj', undefined)).toBeNull()
  expect(absolutePath('/etc/a', undefined, undefined)).toBe('/etc/a')
  expect(absolutePath('C:\\a', undefined, undefined)).toBe('C:\\a')
  expect(absolutePath('a/b', '/proj/', undefined)).toBe('/proj/a/b')
  expect(absolutePath('../a', '/proj', undefined)).toBe('/proj/../a')
  expect(absolutePath('a', undefined, undefined)).toBeNull()
})

test('listText: sections, an empty list, and control characters shown as ?', () => {
  expect(listText([{ kind: 'path', value: '/a' }, { kind: 'name', value: 'out' }])).toBe(
    'Paths:\n  /a\nNames (every project):\n  out',
  )
  expect(listText([{ kind: 'name', value: 'out' }])).toBe('Names (every project):\n  out')
  expect(listText([])).toBe('The allowlist is empty.')
  expect(listText([{ kind: 'path', value: '/a\u001b[31m\nfake' }])).toBe('Paths:\n  /a?[31m?fake')
})

// -- the command, from the person's Enter ---------------------------------------

test('allow <path>: the CLI is called by argv, a relative path made absolute against the session folder', async ($, on) => {
  const w = world(on, () => ADDED)
  expect(await say($, 'allow .env')).toBe('Allowed /proj/app/.env.')
  expect(w.ran).toEqual([[...CLI, 'allow', '--json', '/proj/app/.env']])
  expect(w.read).toEqual(['/home/me/.harpo/install.json'])
})

test('allow <path>: absolute, ~ and quoted paths with spaces reach the CLI as one argument', async ($, on) => {
  const w = world(on, () => ADDED)
  await say($, 'allow /etc/x.env')
  await say($, 'allow ~/notes/x.env')
  await say($, 'allow "My Project/a b.env"')
  await say($, "allow 'it''s here/x'")
  expect(w.ran).toEqual([
    [...CLI, 'allow', '--json', '/etc/x.env'],
    [...CLI, 'allow', '--json', '/home/me/notes/x.env'],
    [...CLI, 'allow', '--json', '/proj/app/My Project/a b.env'],
    [...CLI, 'allow', '--json', '/proj/app/its here/x'],
  ])
})

test('allow <path>: shell metacharacters stay one inert argument', async ($, on) => {
  const w = world(on, () => ADDED)
  await say($, 'allow "/tmp/$(id); rm -rf x | `y`"')
  expect(w.ran).toEqual([[...CLI, 'allow', '--json', '/tmp/$(id); rm -rf x | `y`']])
})

test('allow <path>: exists, and a refusal from the CLI', async ($, on) => {
  const w = world(on, argv =>
    argv.includes('/proj/app/gone')
      ? json({ ok: false, error: '/proj/app/gone: not found', needs_confirmation: false }, 1)
      : json({ ok: true, action: 'exists', kind: 'path', value: '/proj/app/.env' }),
  )
  expect(await say($, 'allow .env')).toBe('/proj/app/.env was already allowed.')
  expect(await say($, 'allow gone')).toBe('Not allowed: /proj/app/gone: not found')
  expect(w.ran.length).toBe(2)
})

test('allow --name: runs with --name and reports the name', async ($, on) => {
  const w = world(on, () => json({ ok: true, action: 'added', kind: 'name', value: 'graphify-out' }))
  expect(await say($, 'allow --name graphify-out')).toBe('Allowed the name graphify-out in every project.')
  expect(w.ran).toEqual([[...CLI, 'allow', '--json', '--name', 'graphify-out']])
  expect(w.asked).toEqual([])
})

const NEEDS = json({ ok: false, error: 'src is a broad name. Re-run with --yes to allow it.', needs_confirmation: true }, 3)

test('allow --name: exit 3 asks, and only "Allow it everywhere" re-runs with --yes', async ($, on) => {
  const w = world(
    on,
    argv => (argv.includes('--yes') ? json({ ok: true, action: 'added', kind: 'name', value: 'src' }) : NEEDS),
    EVERYWHERE,
  )
  expect(await say($, 'allow --name src')).toBe('Allowed the name src in every project.')
  expect(w.asked).toEqual([
    { question: 'Allow the name src in every project? It is broad.', options: [EVERYWHERE, 'Cancel'] },
  ])
  expect(w.ran).toEqual([
    [...CLI, 'allow', '--json', '--name', 'src'],
    [...CLI, 'allow', '--json', '--yes', '--name', 'src'],
  ])
})

test('allow --name: Cancel, free text and a dismissal never re-run', async ($, on) => {
  const w = world(on, () => NEEDS)
  for (const answer of ['Cancel', 'allow it everywhere', 'yes', null]) {
    w.use(() => NEEDS, answer)
    expect(await say($, 'allow --name src')).toBe('Cancelled.')
    expect(w.ran.length).toBe(1)
  }
})

test('--yes typed by the person is not passed through', async ($, on) => {
  const w = world(on, () => NEEDS, 'Cancel')
  expect((await say($, 'allow --name src --yes'))?.startsWith('Usage')).toBe(true)
  expect(w.ran).toEqual([])
})

test('remove: path and name', async ($, on) => {
  const w = world(on, argv =>
    json({ ok: true, action: 'removed', kind: argv.includes('--name') ? 'name' : 'path', value: argv.at(-1) }),
  )
  expect(await say($, 'remove sub/x.env')).toBe('Removed /proj/app/sub/x.env.')
  expect(await say($, 'remove --name out')).toBe('Removed the name out.')
  expect(w.ran).toEqual([
    [...CLI, 'allow', '--remove', '--json', '/proj/app/sub/x.env'],
    [...CLI, 'allow', '--remove', '--json', '--name', 'out'],
  ])
})

test('remove: a refusal reads "Not removed"', async ($, on) => {
  world(on, () => json({ ok: false, error: 'not on the allowlist', needs_confirmation: false }, 1))
  expect(await say($, 'remove x')).toBe('Not removed: not on the allowlist')
})

test('allowlist, list and allow --list show the list', async ($, on) => {
  const w = world(on, () =>
    json({ entries: [{ kind: 'path', value: '/a/.env' }, { kind: 'name', value: 'graphify-out' }] }),
  )
  const want = 'Paths:\n  /a/.env\nNames (every project):\n  graphify-out'
  expect(await say($, 'allowlist')).toBe(want)
  expect(await say($, 'list')).toBe(want)
  expect(await say($, 'allow --list')).toBe(want)
  expect(w.ran).toEqual([
    [...CLI, 'allow', '--list', '--json'],
    [...CLI, 'allow', '--list', '--json'],
    [...CLI, 'allow', '--list', '--json'],
  ])
})

test('an empty list, and text from the CLI is sanitised', async ($, on) => {
  const w = world(on, () => json({ entries: [] }))
  expect(await say($, 'allowlist')).toBe('The allowlist is empty.')
  w.use(() => json({ ok: false, error: 'bad\u001b[31m thing', needs_confirmation: false }, 1))
  expect(await say($, 'allow x')).toBe('Not allowed: bad?[31m thing')
})

test('help, no arguments, unknown words and usage never run the CLI', async ($, on) => {
  const w = world(on, () => ADDED)
  expect(await say($, '')).toBe(HELP)
  expect(await say($, 'help')).toBe(HELP)
  expect(await say($, 'frob\u001bnicate now')).toBe(`Unknown: frob?nicate\n${HELP}`)
  expect((await say($, 'allow'))?.startsWith('Usage')).toBe(true)
  expect(w.ran).toEqual([])
})

// -- failures ----------------------------------------------------------------

test('the CLI is missing: the install message, for a change and for the list', async ($, on) => {
  const w = world(on, () => 'missing')
  expect(await say($, 'allow x')).toBe(NOT_FOUND)
  expect(await say($, 'remove --name x')).toBe(NOT_FOUND)
  expect(await say($, 'allowlist')).toBe(NOT_FOUND)
  expect(w.ran.length).toBe(3)
})

test('an unreadable answer is a safe message, never a throw', async ($, on) => {
  const bad = 'The Harpocrates CLI returned an answer this plugin cannot read.'
  const w = world(on, () => 'missing')
  for (const stdout of ['', 'not json', '[1,2]', 'null', '{"ok":']) {
    w.use(() => ({ exitCode: 0, stdout }))
    expect(await say($, 'allow x')).toBe(bad)
    expect(await say($, 'allowlist')).toBe(bad)
  }
})

test('exit 0 without ok:true, or a non-zero exit with no error text, is a failure', async ($, on) => {
  const w = world(on, () => json({ ok: false }, 0))
  expect(await say($, 'allow x')).toBe('Not allowed: the CLI exited with code 0')
  w.use(() => json({}, 2))
  expect(await say($, 'remove x')).toBe('Not removed: the CLI exited with code 2')
})

// -- the interpreter comes from the install record, never from PATH -------------

const RECORDS: [string, string | null][] = [
  ['no record file', null],
  ['an empty file', ''],
  ['garbage', 'not json {'],
  ['not an object', '[1]'],
  ['no python key', JSON.stringify({ version: 1 })],
  ['a bare name', JSON.stringify({ python: 'python3' })],
  ['a relative path', JSON.stringify({ python: './venv/bin/python' })],
  ['a path with a dot-dot start', JSON.stringify({ python: '../python' })],
  ['an empty string', JSON.stringify({ python: '' })],
  ['a non-string', JSON.stringify({ python: ['/usr/bin/python3'] })],
  ['python_args not a list', JSON.stringify({ python: PY, python_args: '-P' })],
  ['python_args empty', JSON.stringify({ python: PY, python_args: [] })],
  ['python_args with another flag', JSON.stringify({ python: PY, python_args: ['-P', '-c'] })],
  ['python_args that import', JSON.stringify({ python: PY, python_args: ['-m', 'evil'] })],
  ['python_args with a non-string', JSON.stringify({ python: PY, python_args: [1] })],
  ['python_args -E alone, which keeps the cwd on sys.path', JSON.stringify({ python: PY, python_args: ['-E'] })],
  ['python_args -E twice', JSON.stringify({ python: PY, python_args: ['-E', '-E'] })],
  ['python_args -E with another flag', JSON.stringify({ python: PY, python_args: ['-E', '-P', '-c'] })],
  ['python_args null', JSON.stringify({ python: PY, python_args: null })],
  ['a control character', JSON.stringify({ python: '/usr/bin/py\nthon' })],
]

test('without a usable recorded python the install message is shown and nothing runs', async ($, on) => {
  const w = world(on, () => ADDED, EVERYWHERE)
  for (const [, record] of RECORDS) {
    w.setRecord(record)
    for (const args of ['allow x', 'allow --name src', 'remove x', 'remove --name src', 'allowlist']) {
      expect(await say($, args)).toBe(NOT_INSTALLED)
    }
  }
  expect(w.ran).toEqual([])
  expect(w.asked).toEqual([])
})

test('a recorded absolute python runs with its recorded flags, -I when the record has none', async ($, on) => {
  const w = world(on, () => ADDED)
  w.setRecord(JSON.stringify({ version: 1, python: '/opt/py 3/bin/python' }))
  await say($, 'allow /x')
  w.setRecord(JSON.stringify({ python: 'C:\\Python\\python.exe', python_args: ['-I'] }))
  await say($, 'allow /y')
  w.setRecord(JSON.stringify({ python: '/p/python', python_args: ['-P'] }))
  await say($, 'allow /z')
  w.setRecord(JSON.stringify({ python: '/u/python', python_args: ['-E', '-P'] })) // an install in the user site (SEC-08)
  await say($, 'allow /w')
  expect(w.ran).toEqual([
    ['/opt/py 3/bin/python', '-I', '-m', 'Harpocrates.cli', 'allow', '--json', '/x'],
    ['C:\\Python\\python.exe', '-I', '-m', 'Harpocrates.cli', 'allow', '--json', '/y'],
    ['/p/python', '-P', '-m', 'Harpocrates.cli', 'allow', '--json', '/z'],
    ['/u/python', '-E', '-P', '-m', 'Harpocrates.cli', 'allow', '--json', '/w'],
  ])
})

test('no bare harpocrates is ever run, whatever the record says', async ($, on) => {
  const w = world(on, () => ADDED)
  for (const [, record] of RECORDS) {
    w.setRecord(record)
    await say($, 'allow x')
    await say($, 'allowlist')
  }
  expect(w.ran.every(argv => argv[0] === '/venv/bin/python' || argv[0]?.startsWith('/'))).toBe(true)
  expect(w.ran.some(argv => argv[0] === 'harpocrates')).toBe(false)
})

test('the record is read from ~/.harpo first and from ~/.harpocrates when the new folder has none', async ($, on) => {
  const w = world(on, () => ADDED)
  const NEW = '/home/me/.harpo/install.json'
  const OLD = '/home/me/.harpocrates/install.json'
  const newer = JSON.stringify({ python: '/new/python', python_args: ['-P'] })
  const older = JSON.stringify({ python: '/old/python', python_args: ['-P'] })

  w.setRecordAt({ [NEW]: newer, [OLD]: older })
  await say($, 'allow /a')
  expect(w.read).toEqual([NEW])

  w.read.length = 0
  w.setRecordAt({ [OLD]: older })
  await say($, 'allow /b')
  expect(w.read).toEqual([NEW, OLD])

  w.read.length = 0
  w.setRecordAt({ [NEW]: 'not json {', [OLD]: older }) // the new record decides, even when it is unusable
  expect(await say($, 'allow /c')).toBe(NOT_INSTALLED)
  expect(w.read).toEqual([NEW])

  w.setRecordAt({})
  expect(await say($, 'allow /d')).toBe(NOT_INSTALLED)
  expect(w.ran).toEqual([
    ['/new/python', '-P', '-m', 'Harpocrates.cli', 'allow', '--json', '/a'],
    ['/old/python', '-P', '-m', 'Harpocrates.cli', 'allow', '--json', '/b'],
  ])
})

test('help and usage do not read the record', async ($, on) => {
  const w = world(on, () => ADDED)
  await say($, 'help')
  await say($, 'allow')
  expect(w.read).toEqual([])
})

// -- only the person -----------------------------------------------------------

const OTHERS: PromptOrigin[] = [
  { kind: 'plugin', name: 'other' },
  { kind: 'sdk' },
  { kind: 'bridge' },
  { kind: 'unclassified' },
  { kind: 'task-notification' },
  { kind: 'scheduled-trigger' },
  { kind: 'peer' },
  { kind: 'peer-send-message' },
  { kind: 'surprise' } as unknown as PromptOrigin,
]

test('every origin but composer is refused for allow and remove, and no process runs', async ($, on) => {
  const w = world(on, () => ADDED, EVERYWHERE)
  for (const origin of OTHERS) {
    for (const args of [
      'allow x',
      'allow --name src',
      'remove x',
      'remove --name src',
      'ALLOW x',
      '"allow" x',
      'allow',
      'allow a b',
    ]) {
      expect(await say($, args, origin)).toBe(REFUSED)
    }
  }
  expect(w.ran).toEqual([])
  expect(w.asked).toEqual([])
})

test('a run with no origin is refused too', async ($, on) => {
  const w = world(on, () => ADDED)
  const ran = await $.command.run({ command: 'harpo', args: 'allow x', presentation: PRESENTATION } as never)
  expect(ran.text).toBe(REFUSED)
  expect(w.ran).toEqual([])
})

test('help, unknown and the list run for any origin', async ($, on) => {
  const w = world(on, () => json({ entries: [{ kind: 'name', value: 'out' }] }))
  for (const origin of OTHERS) {
    expect(await say($, 'help', origin)).toBe(HELP)
    expect(await say($, 'allowlist', origin)).toBe('Names (every project):\n  out')
    expect(await say($, 'allow --list', origin)).toBe('Names (every project):\n  out')
  }
  expect(w.ran.length).toBe(OTHERS.length * 2)
  expect(w.ran.every(argv => argv.join(' ') === [...CLI, 'allow', '--list', '--json'].join(' '))).toBe(true)
})

test('nothing is put in context for the model', async ($, on) => {
  world(on, () => ADDED)
  const ran = await $.command.run({
    command: 'harpo',
    args: 'allow x',
    origin: { kind: 'composer' },
    presentation: PRESENTATION,
  })
  expect(ran.context).toBeUndefined()
})

test('refused origins do not even read the record', async ($, on) => {
  const w = world(on, () => ADDED)
  expect(await say($, 'allow x', { kind: 'sdk' })).toBe(REFUSED)
  expect(w.read).toEqual([])
})

test('changeArgv: the argument vectors', () => {
  expect(changeArgv('add', { kind: 'path', value: '/a' })).toEqual(['allow', '--json', '/a'])
  expect(changeArgv('add', { kind: 'name', value: 'n' }, true)).toEqual(['allow', '--json', '--yes', '--name', 'n'])
  expect(changeArgv('remove', { kind: 'name', value: 'n' })).toEqual(['allow', '--remove', '--json', '--name', 'n'])
})

// -- the old name: /harpocrates is /harpo (FR-NAME-02) ----------------------------

test('session start registers /harpo and, for two more minor releases, /harpocrates', async ($, on) => {
  const registered: string[] = []
  on('command.register', (_$, e) => {
    registered.push(e.name)
    return { value: undefined }
  })
  on('session.start', (_$, e) => e) // the engine beneath the plugin: it just starts
  await $.session.start({ cwd: '/proj/app' } as never)
  expect(registered).toEqual(['harpo', 'harpocrates'])
})

for (const command of NAMES) {
  test(`/${command}: allow, remove, list and help run the same CLI calls and print the same text`, async ($, on) => {
    const w = world(on, argv =>
      argv.includes('--list') ? json({ entries: [{ kind: 'name', value: 'out' }] }) : ADDED,
    )
    expect(await say($, 'allow .env', { kind: 'composer' }, command)).toBe('Allowed /proj/app/.env.')
    expect(await say($, 'remove .env', { kind: 'composer' }, command)).toBe('Removed /proj/app/.env.')
    expect(await say($, 'allowlist', { kind: 'composer' }, command)).toBe('Names (every project):\n  out')
    expect(await say($, 'help', { kind: 'composer' }, command)).toBe(HELP)
    expect(await say($, 'frobnicate', { kind: 'composer' }, command)).toBe(`Unknown: frobnicate\n${HELP}`)
    expect(w.ran).toEqual([
      [...CLI, 'allow', '--json', '/proj/app/.env'],
      [...CLI, 'allow', '--remove', '--json', '/proj/app/.env'],
      [...CLI, 'allow', '--list', '--json'],
    ])
  })

  test(`/${command}: a broad name asks, and only the exact Allow it everywhere re-runs with --yes`, async ($, on) => {
    const w = world(
      on,
      argv =>
        argv.includes('--yes')
          ? json({ ok: true, action: 'added', kind: 'name', value: 'src' })
          : json({ ok: false, error: 'src is broad', needs_confirmation: true }, 3),
      EVERYWHERE,
    )
    expect(await say($, 'allow --name src', { kind: 'composer' }, command)).toBe('Allowed the name src in every project.')
    expect(w.asked.length).toBe(1)
    expect(w.ran.length).toBe(2)
    expect(w.ran[1]).toEqual([...CLI, 'allow', '--json', '--yes', '--name', 'src'])
  })

  test(`/${command}: every origin but composer is refused for a change, and no process runs`, async ($, on) => {
    const w = world(on, () => ADDED, EVERYWHERE)
    for (const origin of OTHERS) {
      for (const args of ['allow x', 'allow --name src', 'remove x', 'ALLOW x', '"allow" x', 'allow', 'allow a b']) {
        expect(await say($, args, origin, command)).toBe(REFUSED)
      }
    }
    const noOrigin = await $.command.run({ command, args: 'allow x', presentation: PRESENTATION } as never)
    expect(noOrigin.text).toBe(REFUSED)
    expect(w.ran).toEqual([])
    expect(w.asked).toEqual([])
    expect(w.read).toEqual([])
  })

  test(`/${command}: help, unknown and the list run for any origin; nothing goes into context`, async ($, on) => {
    const w = world(on, () => json({ entries: [{ kind: 'name', value: 'out' }] }))
    for (const origin of OTHERS) {
      expect(await say($, 'help', origin, command)).toBe(HELP)
      expect(await say($, 'allowlist', origin, command)).toBe('Names (every project):\n  out')
    }
    expect(w.ran.length).toBe(OTHERS.length)
    const ran = await $.command.run({ command, args: 'help', origin: { kind: 'composer' }, presentation: PRESENTATION })
    expect(ran.context).toBeUndefined()
  })
}

test('the help and the refusal name /harpo, the command to type', () => {
  expect(HELP).toContain('/harpo allow <path>')
  expect(HELP).not.toContain('/harpocrates')
  expect(REFUSED).toContain('type /harpo in')
  expect(NOT_INSTALLED).toContain('`harpo install`')
})

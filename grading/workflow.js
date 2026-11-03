export const meta = {
  name: 'grade-zero',
  description: 'Grade Zero eval replies with grading/rubric.md: one Claude Haiku call per section of each model',
  whenToUse: 'After `uv run grading/grade.py make NAME ...`; args = the "grades/NAME:SECTIONS" strings it prints',
  phases: [{ title: 'Grade' }],
}
// args: "grades/NAME:casual+task+identity+world+battery" (paths relative to the repo root). One grader per section file
// grades/NAME/sec_<section>.jsonl, writing grades/NAME/labels_<section>.json; each grader reads grading/rubric.md itself.
// The graders run on Claude Haiku ('haiku'; our reference grades used Claude Haiku 5.5).
const jobs = []
for (const a of args) {
  const [d, secs] = a.split(':')
  for (const s of (secs ? secs.split('+') : ['casual', 'task', 'identity', 'world', 'battery'])) jobs.push({ d, s })
}
const res = await parallel(jobs.map(({ d, s }) => () =>
  agent(`Paths below without a leading / are relative to your current working directory (the repository root). Read the grading instructions in grading/rubric.md and follow them exactly. In those instructions, BATCH is ${d}/sec_${s}.jsonl and OUT is ${d}/labels_${s}.json.`,
        { label: `${d.split('/').pop()} ${s}`, phase: 'Grade', model: 'haiku' })))
return jobs.map((j, i) => `${j.d.split('/').pop()}/${j.s}: ${res[i]}`)

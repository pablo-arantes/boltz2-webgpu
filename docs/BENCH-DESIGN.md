# bench.html — the design plan, and the review that changed it

`screen.html` works and keeps working; this is a second front end over the same
controller, not a replacement. The plan was written before the code, reviewed
against the brief, revised, and only then built — the two passes the
`frontend-design` skill asks for. What follows is that record, including the
parts I threw away, because the discarded options are the argument for the ones
that shipped.

## The brief, stated rather than assumed

Nobody wrote a brief, so here is the one I designed against.

**Subject.** A screening bench that runs in a browser tab. One protein, up to a
few hundred small molecules, co-folded and scored for affinity on the reader's
own GPU. No server, no account, no upload.

**Audience.** Structural biologists and medicinal/computational chemists. People
who know what an IC50 is, and who are *right* to distrust an affinity number
produced by a web page in two seconds.

**Primary job.** Rank a set of molecules against one target, see each pose in
the pocket, and never let a ranking be mistaken for a potency. That last clause
is not decoration: this page has two affinity outputs that must stay labelled
and separate, an ensemble spread that must stay visible, and a sampler running
far below upstream's settings. A design that makes the numbers look more
confident than they are would be a worse design however handsome.

## Where the look comes from

Not from dashboards. From the bench: **a virtual screen is a plate.** One
target, N compounds, one number per well. That is the oldest object in this
subject's world and it is still how a screen is read — eyes down the plate for
the wells that lit up, then back to the ones that matter.

So the plate is the page's one bold object, and it does three jobs at once that
were previously done by three different widgets: it shows progress while the
screen runs, it shows every score at a glance when it finishes, and it is how
you navigate — click a well, see that molecule alone in the pocket.

## Tokens

### Colour — six base values

| token | hex | what it is |
| --- | --- | --- |
| `--paper` | `#F4F6F7` | the page ground: a cool off-white, the colour of bench epoxy |
| `--surface` | `#FFFFFF` | anything you type into |
| `--ink` | `#152430` | text: a deep slate-navy, a hue rather than a stand-in for black |
| `--quiet` | `#5E7180` | secondary text and axis labels |
| `--rule` | `#CBD5DA` | 1px separators, used only between different *kinds* of thing |
| `--carrier` | `#11212B` | the plate's dark carrier — the one dark object on the page |

The measurement ramp is **not** part of the palette, and that is the point.
`--cool #3156B8` → `--warm #D66C14` is lifted unchanged from
`web/screen-scene.js`, where it was chosen to be colour-blind safe and to read
as cool/warm in greyscale. It is the only saturated colour on the page, and it
means score and nothing else: the same two ends colour the wells, the scatter's
points, the table's swatches and the atoms in the 3D scene. A reader who learns
one ramp has learned the whole page. `--fault #9E2B20` is the only other
chromatic value, and it appears only on a molecule that failed.

### Type — two faces, one rule

| role | stack | used for |
| --- | --- | --- |
| language | `"Avenir Next", Avenir, "Segoe UI Variable Text", Candara, "Trebuchet MS", system-ui, sans-serif` | headings, labels, prose, buttons |
| measured | `"SF Mono", Menlo, "Cascadia Mono", Consolas, ui-monospace, monospace` | every number, sequence, SMILES and machine readout |

The rule is the personality: **if it is measured, it is monospaced; if it is
language, it is not.** A sequence is monospaced because residues line up; a
score is monospaced with `tabular-nums` because a column of numbers must be
comparable down the column. The same face in a *label* would be costume, so
labels stay in the sans.

Scale, 1.25 off a 16px body: 12.8 / 16 / 20 / 25 / 31.25px. Headline at 31.25px,
weight 600, tracking −0.02em, sentence case. Body 16/1.5, measure capped at
68ch. No all-caps anywhere.

**No web fonts.** Both stacks are system faces, and that is a constraint rather
than a preference: this page's whole claim is that nothing leaves the machine
except an optional MSA search, and a Google Fonts link would send every
reader's IP address to Google on load. A licensed display face would have to be
self-hosted and carry its licence into the repository. The personality comes
from the treatment instead — scale, weight, the mono/sans rule — which is
cheaper and does not undo the page's one promise.

### Layout

Left-aligned throughout; a lab readout is not centred. Numeric table columns
right-aligned with tabular figures so magnitudes line up. 1180px measure for
the stage, with the plate bleeding the full width.

```
┌──────────────────────────────────────────────────────────────────┐
│ Boltz-2 virtual screening             [Run screen] [Stop] [ZIP]  │  paper
│ One protein, many molecules, folded and scored on your own GPU.  │
├──────────────────────────────────────────────────────────────────┤
│ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  │  carrier
│ one well per molecule · fills when it scores · click to isolate  │  ← mono readout
├───────────────────────────────┬──────────────────────────────────┤
│ Target protein         33 aa  │ Molecules to screen        27    │
│ [MKTAYIAKQRQISFVKSHFSR…    ]  │ [CCO                        ]    │  surface
│ Alignment  [single sequence]  │ or choose a .smi file            │
├───────────────────────────────┴──────────────────────────────────┤
│ seed 0    recycles 1    steps 12    samples 1                    │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│                  3D structure, the full measure                  │
│                                                                  │
│ Show [all molecules]   Colour [binding probability]              │
│ target drawn once; 27 of 27 superposed; ramp 0.057 → 0.384       │
├─────────────────────────────┬────────────────────────────────────┤
│ scatter                     │ ranked molecules                   │
│ (2/5)                       │ (3/5)                              │
├─────────────────────────────┴────────────────────────────────────┤
│ What to distrust about these numbers (prose, 68ch)               │
└──────────────────────────────────────────────────────────────────┘
```

Two layouts were drawn and rejected.

**A left setup rail, stage right.** Rejected on measurement, not taste: the
viewer's canvas has to earn its height with width, and `web/screen.css`'s own
header records four failed attempts at getting it past 600px of a 948px column.
A permanent 280px rail spends that width on two text boxes you touch once.

**Setup as a collapsible drawer.** Rejected because it hides the two fields
that are the entire input while leaving a chevron that has to be explained. The
inputs are short; they can simply be small.

### Principles

1. **The plate is the page.** Progress, score and navigation in one object.
2. **One ramp, everywhere.** Colour means score. Nothing else is saturated.
3. **Every number next to its scale.** No bare figure: units, domain, or the
   ramp's two ends, so a colour or a value cannot be read as more than it is.
4. **Two faces, one rule.** Measured in mono, language in sans.
5. **Quiet elsewhere.** No shadows, no second accent, no rounded identical
   cards. One moment of motion: a well taking its colour as its result lands.

## The review, and what it changed

Three parts of the first plan were the thing I would have produced for any
dashboard brief, so they were revised before any code was written.

**A dark header bar.** My first sketch put the header and the plate together on
the dark carrier. That is a toolbar, and it would have made the dark ground
*chrome* rather than *data*. Revised: the header sits on paper and the carrier
is the plate alone, so the single dark object on the page is the one that holds
the measurements. The boldness is spent on the data, which is the point of
spending it in one place.

**Status as a hero number.** The obvious move is "0.384" at 64px with a label
under it. It was rejected on the brief: the single biggest risk in this page is
a reader over-trusting a number, and a hero figure is an argument for
confidence. The hero is the plate — a *distribution* — which says "here is a
ranking" rather than "here is a potency".

**Numbered section markers.** Drafted 01 / 02 / 03 down the setup column, then
cut: a screen's parts are not a sequence. You can set molecules before the
target, and you will change the molecules and re-run without touching anything
else. Numbering would have encoded an order the work does not have.

Two further defaults were never in the plan but are worth naming, because the
page being replaced has them: an ALL-CAPS eyebrow over every section (`TARGET`,
`LIGANDS`, `SEED`, `QUEUE`, `SCENE`, `RANKED`), and six identically rounded
grey cards stacked down the page. Both are in the skill's list of tells, and
both are what made the old page read as generic however well it worked.

## What this page does NOT change

The controller. `bench.html` loads the same `web/screen.js` as `screen.html`,
satisfies the same twenty-two element ids, and carries py2Dmol's viewer subtree
verbatim. Nothing about the run loop, the weights, the MSA cache, the
superposition, the two affinity heads or the ZIP is duplicated.

That is deliberate, and it is the opposite of what a "new front end" usually
means. A forked controller would be 853 lines of run loop, device handling and
persistence copied for a layout change, and the copy would drift — the two
pages would disagree about what a screen does within a month. The honest
version of "keep the old page as a backup" is a page whose *markup and
stylesheet* nothing touched, which is exactly what `screen.html` and
`web/screen.css` now are.

The one shared change is additive: `renderQueue` now puts the ligand's id and
its score colour on each chip and makes it clickable. `web/screen.css` does not
read either, so `screen.html` looks exactly as it did; `web/bench.css` reads
both, and that is what turns a row of chips into a plate.

## The critique round, which a screenshot settled

The skill's advice to take screenshots and critique your own work paid for
itself four times. What the first build got wrong, and what the picture showed:

**The measurement was a hairline.** Each well had a 3px rule of score colour
over a 100px-wide molecule id. The id - `ligand_0007`, the least interesting
thing in the cell - was the whole object, and the score, which the plate exists
to show, was almost invisible. The band is now 12px and the dominant half of the
well; the name sits under it, smaller and quieter. This was the single biggest
improvement in the pass, and it was the one thing the plan got exactly backwards
about where the boldness goes.

**Two thirds of the viewer was white.** 600px of canvas at 1130 wide, with
py2Dmol fitting its camera to one small protein. Set to 480, which keeps the
full width - a complex reads landscape - and loses the height nothing was in.

**Setting that height did nothing, and the probe said why.** `canvas.height`
still read 600 after `#canvasContainer { height: 480px }`, because the vendor
pins `.py2dmol-slot--1 > .py2dmol-slot-body` to 600x600 and sizes the canvas
from THAT box. The same trap as the width, which web/screen.css's header records
costing four attempts. Both boxes are set now.

**The viewer's own buttons were a different product.** Six grey pills in a
rounded tray with a soft shadow, in py2Dmol's palette, sitting directly under
the canvas. Only colour, border and radius are overridden; the vendor's panel
builds and positions itself at runtime and its layout is untouched.

And one thing the plan did not anticipate at all: `<table id="table"></table>`
is not enough. `drawTable` does
`table.querySelector("thead").replaceChildren()` with no guard, so the page
died on "Loading&hellip;" with all twenty-two ids present and a single console
line. Finding it cost a twelve-minute probe run and a direct read of the live
DOM; `test/py2dmol-scaffold.test.js` now checks the `thead`, the `tbody` and the
scatter's viewBox statically, for this page and the next one.

## For the next pass

Things considered and left undone, so they are not rediscovered:

- **The wells are not keyboard-reachable.** A hundred of them in the tab order
  is a trap, and a roving-tabindex grid would add no capability, because
  `#show-ligand` is a native select listing every molecule and doing exactly the
  same thing. If the plate ever gains an action the select does not have, this
  has to change.
- **The plate is in submission order, not rank order.** That is deliberate -
  position is identity on a plate, and the table is what sorts - but a
  "sort the plate" control is a reasonable ask.
- **The scatter's own type is 9-10px in a 440x240 viewBox** and scales with its
  column, so it is small in a 6/13 column. Changing it means changing
  `web/screen-view.js`, which both pages share.

**And the pills took three attempts, two of which looked identical.** The
viewer's six camera controls stayed in py2Dmol's grey through a rule on
`#controlsContainer button` (five of the six are `label.btn-toggle`, not
`button`) and then through a rule on `#controlsContainer .btn-toggle > span`,
which matched nothing: they live in `#navigationControls`, inside
`#rightPanelsContainer`, a **sibling** of the controls tray. Two screenshots
apart, both looking unchanged, and the thing that settled it in one minute was
asking the page for a computed style instead - `null` for the pill while the
real button already read `rgb(255,255,255)`, which says the override works and
the selector is pointed at the wrong box. Scoped to `#viewer-container`, it
also outranks the vendor's `[aria-pressed="true"]` rule, so the latched state
had to be rewritten in this page's own ink or Orient, Focus and Rotate would
have looked permanently off.

🔴 **A screenshot cannot tell "the rule is wrong" from "the rule did not
match".** When a style change appears to do nothing, read the computed value
before changing the rule again.

#import "@preview/cetz:0.4.2"

#set page(width: 22in, height: auto, margin: 0.4in, fill: white)
#set text(font: "Arial", fill: rgb("#000000"))

#let manifest = json("evidence/manifest.json")
#let categories = (
  "tiny_misses",
  "localization_below_0_5",
  "duplicate_hypotheses",
  "radar_supported_clutter_shoreline",
  "ambiguous_omitted_objects",
  "method_specific_recovery_disagreement",
)

#let title(category) = category.replace("_", " ")

#let evidence-image(path) = cetz.canvas({
  import cetz.draw: *
  content(
    (0, 0),
    (16, 10),
    image("evidence/" + path, width: 100%, height: 100%, fit: "cover"),
  )
  rect(
    (0, 0),
    (16, 10),
    stroke: (paint: rgb("#73000A"), thickness: 0.04),
  )
})

#align(center)[
  #text(size: 24pt, weight: "bold", fill: rgb("#73000A"))[
    DETECTION ERROR EVIDENCE QA
  ]
]

#for category in categories {
  let items = manifest.categories.at(category)
  block(above: 0.22in, below: 0.08in)[
    #text(size: 14pt, weight: "bold", fill: rgb("#73000A"))[
      #title(category)
    ]
  ]
  grid(
    columns: (1fr, 1fr, 1fr),
    gutter: 0.14in,
    ..items.map(item => evidence-image(item.evidence_image)),
  )
}

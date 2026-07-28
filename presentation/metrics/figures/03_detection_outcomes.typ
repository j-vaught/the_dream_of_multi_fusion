#import "common.typ": *

#let rows = csv("../data/figure_03_detection_outcomes.csv", row-type: dictionary)
#let methods = (
  "vision_only_tiled",
  "radar_confidence_gated",
  "radar_bounded_crops",
)
#let maximum = 3000
#let fallback-used = rows.any(row => row.detail_available == "False")

#chart-page(
  canvas(length: 1.35cm, {
    import draw: *

    let x0 = 4.2
    let y0 = 1.55
    let width = 13.25
    let bar-height = 1.15
    let gap = 1.35

    for tick in range(0, 7) {
      let count = tick * 500
      let x = x0 + count / maximum * width
      line((x, y0), (x, y0 + 6.35), stroke: grey30 + 0.35pt)
      content(
        (x, y0 - 0.18),
        text(size: 8.5pt, fill: grey70)[#str(count)],
        anchor: "north",
      )
    }
    line((x0, y0), (x0 + width, y0), stroke: black + 0.7pt)

    for (index, method) in methods.enumerate() {
      let row = rows.find(row => row.method == method)
      let y = y0 + 1.05 + (2 - index) * gap
      let tp = int(row.tp)
      let fn = int(row.fn)
      let fp = int(row.fp)
      let tp-end = x0 + tp / maximum * width
      let fn-end = tp-end + fn / maximum * width
      let fp-end = fn-end + fp / maximum * width

      rect(
        (x0, y),
        (tp-end, y + bar-height),
        fill: garnet,
        stroke: black + 0.4pt,
      )
      rect(
        (tp-end, y),
        (fn-end, y + bar-height),
        fill: grey50,
        stroke: black + 0.4pt,
      )
      rect(
        (fn-end, y),
        (fp-end, y + bar-height),
        fill: atlantic,
        stroke: black + 0.4pt,
      )
      content(
        (x0 - 0.25, y + bar-height / 2),
        text(size: 9.5pt, weight: "bold")[
          #method-label(
            method,
            fallback-mark: row.detail_available == "False",
          )
        ],
        anchor: "east",
      )
      if tp > 120 {
        content(
          ((x0 + tp-end) / 2, y + bar-height / 2),
          text(size: 9pt, weight: "bold", fill: white)[TP #str(tp)],
          anchor: "center",
        )
      }
      content(
        ((tp-end + fn-end) / 2, y + bar-height / 2),
        text(size: 9pt, weight: "bold", fill: black)[FN #str(fn)],
        anchor: "center",
      )
      if fp >= 80 {
        content(
          ((fn-end + fp-end) / 2, y + bar-height / 2),
          text(size: 8.5pt, weight: "bold", fill: white)[FP #str(fp)],
          anchor: "center",
        )
      } else {
        content(
          (fp-end + 0.12, y + bar-height / 2),
          text(size: 8.5pt, weight: "bold", fill: atlantic)[FP #str(fp)],
          anchor: "west",
        )
      }
    }

    legend-item((5.3, 9.15), garnet, "True positive")
    legend-item((9.05, 9.15), grey50, "False negative")
    legend-item((13.15, 9.15), atlantic, "False positive")
    content(
      (17.45, 0.42),
      text(size: 8.5pt, fill: grey70)[
        Counts across 300 frames
        #if fallback-used [ · \* aggregate fallback]
      ],
      anchor: "east",
    )
  }),
)

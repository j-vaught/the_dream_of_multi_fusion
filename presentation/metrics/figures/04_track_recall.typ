#import "common.typ": *

#let rows = csv("../data/figure_04_track_recall.csv", row-type: dictionary)
#let has-bounded = rows.any(row => row.method == "radar_bounded_crops")
#let methods = if has-bounded {
  (
    "vision_only_tiled",
    "radar_confidence_gated",
    "radar_bounded_crops",
  )
} else {
  ("vision_only_tiled", "radar_confidence_gated")
}
#let tracks = ("boat", "b1", "b2", "b3", "b4", "b5", "b6", "b7", "b8")

#let heat-color(value) = if value >= 0.9 {
  garnet
} else if value >= 0.5 {
  rose
} else if value > 0 {
  sandstorm
} else {
  grey10
}

#let heat-text(value) = if value >= 0.5 { white } else { black }
#let detail-note = if has-bounded {
  [Recall across all labeled track frames]
} else {
  [Radar-bounded track detail unavailable]
}

#chart-page(
  canvas(length: 1.35cm, {
    import draw: *

    let x0 = 4.35
    let y0 = 2.15
    let cell-width = 1.415
    let cell-height = if has-bounded { 1.42 } else { 2.05 }
    let row-gap = if has-bounded { 0.25 } else { 0.36 }

    for (track-index, track) in tracks.enumerate() {
      let x = x0 + track-index * cell-width
      content(
        (
          x + cell-width / 2,
          y0 + methods.len() * (cell-height + row-gap) + 0.23,
        ),
        text(size: 10pt, weight: "bold")[#track],
        anchor: "south",
      )
      if track == "b2" {
        content(
          (
            x + cell-width / 2,
            y0 + methods.len() * (cell-height + row-gap) - 0.04,
          ),
          text(size: 7.5pt, fill: grey70)[n = 92],
          anchor: "south",
        )
      }
    }

    for (method-index, method) in methods.enumerate() {
      let y = y0 + (methods.len() - 1 - method-index) * (cell-height + row-gap)
      content(
        (x0 - 0.3, y + cell-height / 2),
        text(size: 9.5pt, weight: "bold")[#method-label(method)],
        anchor: "east",
      )
      for (track-index, track) in tracks.enumerate() {
        let row = rows.find(
          row => row.method == method and row.track_id == track,
        )
        let value = float(row.recall)
        let x = x0 + track-index * cell-width
        rect(
          (x, y),
          (x + cell-width, y + cell-height),
          fill: heat-color(value),
          stroke: black + 0.55pt,
        )
        content(
          (x + cell-width / 2, y + cell-height / 2),
          text(
            size: 10pt,
            weight: "bold",
            fill: heat-text(value),
          )[#percent(value)],
          anchor: "center",
        )
      }
    }

    legend-item((5.6, 1.2), grey10, "0%")
    legend-item((8.2, 1.2), sandstorm, "1–49%")
    legend-item((11.25, 1.2), rose, "50–89%")
    legend-item((14.45, 1.2), garnet, "90–100%")
    content(
      (17.1, 0.48),
      text(size: 8.5pt, fill: grey70)[#detail-note],
      anchor: "east",
    )
  }),
)

#import "common.typ": *

#let rows = csv(
  "../data/figure_06_cumulative_false_positives.csv",
  row-type: dictionary,
)
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
#let frame-ticks = (119, 179, 239, 299, 359, 418)
#let y-maximum = if has-bounded { 350 } else { 8 }
#let y-tick-count = if has-bounded { 7 } else { 8 }
#let y-step = if has-bounded { 50 } else { 1 }

#chart-page(
  canvas(length: 1.35cm, {
    import draw: *

    let x0 = 1.75
    let y0 = 1.35
    let width = 15.7
    let height = 7.1

    for tick in range(0, y-tick-count + 1) {
      let value = tick * y-step
      let y = y0 + value / y-maximum * height
      line((x0, y), (x0 + width, y), stroke: grey30 + 0.35pt)
      content(
        (x0 - 0.2, y),
        text(size: 8.5pt, fill: grey70)[#str(value)],
        anchor: "east",
      )
    }
    for frame in frame-ticks {
      let x = x0 + (frame - 119) / 299 * width
      line((x, y0), (x, y0 + height), stroke: grey10 + 0.35pt)
      content(
        (x, y0 - 0.2),
        text(size: 8.5pt, fill: grey70)[#str(frame)],
        anchor: "north",
      )
    }
    line((x0, y0), (x0, y0 + height), stroke: black + 0.7pt)
    line((x0, y0), (x0 + width, y0), stroke: black + 0.7pt)

    for method in methods {
      let selected = rows.filter(row => row.method == method)
      let points = selected.map(row => (
        x0 + (int(row.camera_frame) - 119) / 299 * width,
        y0 + int(row.cumulative_fp) / y-maximum * height,
      ))
      for index in range(1, points.len()) {
        line(
          points.at(index - 1),
          points.at(index),
          stroke: method-color(method) + 1.35pt,
        )
      }
      for index in range(0, points.len(), step: 30) {
        square-marker(points.at(index), method-color(method), size: 0.075)
      }
      let final-row = selected.last()
      let final-point = points.last()
      let label-offset = if method == "radar_confidence_gated" {
        0.35
      } else if method == "vision_only_tiled" {
        0.08
      } else {
        0.15
      }
      content(
        (final-point.at(0) - 0.12, final-point.at(1) + label-offset),
        text(
          size: 9pt,
          weight: "bold",
          fill: method-color(method),
        )[#str(final-row.cumulative_fp)],
        anchor: "south-east",
      )
    }

    legend-item((5.5, 9.12), garnet, "Vision only")
    legend-item((9.55, 9.12), atlantic, "Confidence gated")
    if has-bounded {
      legend-item((14.1, 9.12), black, "Radar bounded")
    }
    content(
      (17.45, 0.42),
      text(size: 8.5pt, fill: grey70)[
        Cumulative false positives
        #if not has-bounded [ · Radar-bounded detail unavailable]
      ],
      anchor: "east",
    )
  }),
)

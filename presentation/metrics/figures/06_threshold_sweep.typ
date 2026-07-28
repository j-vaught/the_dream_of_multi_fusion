#import "common.typ": *

#let rows = csv("../data/figure_06_threshold_sweep.csv", row-type: dictionary)
#let metrics = ("precision", "recall", "f1")
#let selected-row = rows.find(row => row.selected == "True")
#let y-minimum = 0.20
#let y-maximum = 1.00

#chart-page(
  canvas(length: 1.35cm, {
    import draw: *

    let x0 = 1.75
    let y0 = 1.35
    let width = 15.7
    let height = 7.1

    for tick in range(0, 9) {
      let value = y-minimum + tick * 0.10
      let y = y0 + (value - y-minimum) / (y-maximum - y-minimum) * height
      line((x0, y), (x0 + width, y), stroke: grey30 + 0.35pt)
      content(
        (x0 - 0.2, y),
        text(size: 8.5pt, fill: grey70)[#percent(value)],
        anchor: "east",
      )
    }

    for row in rows {
      let threshold = float(row.threshold)
      let x = x0 + (threshold - 0.10) / 0.08 * width
      line((x, y0), (x, y0 + height), stroke: grey10 + 0.35pt)
      content(
        (x, y0 - 0.2),
        text(size: 8.5pt, fill: grey70)[#row.threshold_label],
        anchor: "north",
      )
    }

    let selected-x = x0 + (float(selected-row.threshold) - 0.10) / 0.08 * width
    line(
      (selected-x, y0),
      (selected-x, y0 + height),
      stroke: grey70 + 1.0pt,
    )
    content(
      (selected-x, y0 + height + 0.2),
      text(size: 8.5pt, weight: "bold", fill: grey70)[Selected 0.16],
      anchor: "south",
    )

    line((x0, y0), (x0, y0 + height), stroke: black + 0.7pt)
    line((x0, y0), (x0 + width, y0), stroke: black + 0.7pt)

    for metric in metrics {
      let points = rows.map(row => (
        x0 + (float(row.threshold) - 0.10) / 0.08 * width,
        y0
          + (float(row.at(metric)) - y-minimum)
            / (y-maximum - y-minimum)
            * height,
      ))
      for index in range(1, points.len()) {
        line(
          points.at(index - 1),
          points.at(index),
          stroke: metric-color(metric) + 1.35pt,
        )
      }
      for point in points {
        if metric == "recall" {
          rect(
            (point.at(0) - 0.075, point.at(1) - 0.075),
            (point.at(0) + 0.075, point.at(1) + 0.075),
            fill: white,
            stroke: metric-color(metric) + 0.8pt,
          )
        } else {
          square-marker(point, metric-color(metric), size: 0.075)
        }
      }
    }

    for metric in metrics {
      let value = float(selected-row.at(metric))
      let y = (
        y0
          + (value - y-minimum)
            / (y-maximum - y-minimum)
            * height
      )
      let offset = if metric == "precision" {
        -0.18
      } else {
        0.18
      }
      content(
        (selected-x + offset, y + 0.14),
        text(
          size: 8.5pt,
          weight: "bold",
          fill: metric-color(metric),
        )[#percent(value)],
        anchor: if offset < 0 { "south-east" } else { "south-west" },
      )
    }

    legend-item((5.05, 9.12), garnet, "Precision")
    legend-item((8.25, 9.12), atlantic, "Recall")
    legend-item((11.05, 9.12), black, "F1")
    content(
      (17.45, 0.42),
      text(size: 8.5pt, fill: grey70)[
        Validation split · focused 20–100% scale
      ],
      anchor: "east",
    )
  }),
)

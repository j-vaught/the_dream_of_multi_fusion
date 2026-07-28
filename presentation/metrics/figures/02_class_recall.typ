#import "common.typ": *

#let rows = csv("../data/figure_02_class_recall.csv", row-type: dictionary)
#let methods = (
  "vision_only_tiled",
  "radar_confidence_gated",
  "radar_bounded_crops",
)
#let classes = ("boat", "buoy")
#let fallback-used = rows.any(row => row.detail_available == "False")

#chart-page(
  canvas(length: 1.35cm, {
    import draw: *

    let x0 = 1.75
    let y0 = 1.35
    let width = 15.7
    let height = 7.25
    let group-width = width / 3
    let bar-width = 1.05

    for tick in range(0, 6) {
      let value = tick / 5
      let y = y0 + value * height
      line((x0, y), (x0 + width, y), stroke: grey30 + 0.35pt)
      content(
        (x0 - 0.22, y),
        text(size: 9pt, fill: grey70)[#percent(value)],
        anchor: "east",
      )
    }
    line((x0, y0), (x0, y0 + height), stroke: black + 0.7pt)
    line((x0, y0), (x0 + width, y0), stroke: black + 0.7pt)

    for (method-index, method) in methods.enumerate() {
      let center = x0 + (method-index + 0.5) * group-width
      let method-row = rows.find(
        row => row.method == method and row.class_name == "boat",
      )
      for (class-index, class-name) in classes.enumerate() {
        let row = rows.find(
          row => row.method == method and row.class_name == class-name,
        )
        let value = float(row.recall)
        let left = center + (class-index - 0.5) * 1.42 - bar-width / 2
        let fill-color = if class-name == "boat" { garnet } else { atlantic }
        rect(
          (left, y0),
          (left + bar-width, y0 + value * height),
          fill: fill-color,
          stroke: black + 0.45pt,
        )
        content(
          (left + bar-width / 2, y0 + value * height + 0.18),
          text(size: 9pt, weight: "bold")[#percent(value)],
          anchor: "south",
        )
      }
      content(
        (center, y0 - 0.32),
        text(size: 10pt, weight: "bold")[
          #method-label(
            method,
            fallback-mark: method-row.detail_available == "False",
          )
        ],
        anchor: "north",
      )
    }

    legend-item((6.25, 9.22), garnet, "Boat")
    legend-item((9.55, 9.22), atlantic, "Buoy")
    if fallback-used {
      content(
        (17.45, 0.42),
        text(size: 8.5pt, fill: grey70)[\* Aggregate fallback],
        anchor: "east",
      )
    }
  }),
)

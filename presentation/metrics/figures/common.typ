#import "@preview/cetz:0.4.2": canvas, draw

#let garnet = rgb("#73000A")
#let black = rgb("#000000")
#let white = rgb("#FFFFFF")
#let charcoal = rgb("#363636")
#let grey70 = rgb("#5C5C5C")
#let grey50 = rgb("#A2A2A2")
#let grey30 = rgb("#C7C7C7")
#let grey10 = rgb("#ECECEC")
#let sandstorm = rgb("#FFF2E3")
#let rose = rgb("#CC2E40")
#let atlantic = rgb("#466A9F")
#let congaree = rgb("#1F414D")
#let horseshoe = rgb("#65780B")
#let honeycomb = rgb("#A49137")

#let chart-page(body) = {
  set page(width: 10in, height: 5.625in, margin: 0pt, fill: white)
  set text(font: "Calibri", size: 11pt, fill: charcoal)
  body
}

#let method-label(key, fallback-mark: false) = {
  let label = if key == "vision_only_tiled" {
    "Vision only"
  } else if key == "radar_confidence_gated" {
    "Confidence gated"
  } else {
    "Radar bounded"
  }
  if fallback-mark and key == "radar_bounded_crops" {
    label + "*"
  } else {
    label
  }
}

#let method-color(key) = if key == "vision_only_tiled" {
  garnet
} else if key == "radar_confidence_gated" {
  atlantic
} else {
  black
}

#let metric-label(key) = if key == "precision" {
  "Precision"
} else if key == "recall" {
  "Recall"
} else {
  "F1"
}

#let metric-color(key) = if key == "precision" {
  garnet
} else if key == "recall" {
  atlantic
} else {
  black
}

#let percent(value) = str(calc.round(value * 100, digits: 1)) + "%"

#let legend-item(position, color, label) = {
  draw.rect(
    position,
    (position.at(0) + 0.34, position.at(1) + 0.24),
    fill: color,
    stroke: black + 0.35pt,
  )
  draw.content(
    (position.at(0) + 0.46, position.at(1) + 0.12),
    text(size: 9.5pt, fill: charcoal)[#label],
    anchor: "west",
  )
}

#let square-marker(position, color, size: 0.09) = {
  draw.rect(
    (position.at(0) - size, position.at(1) - size),
    (position.at(0) + size, position.at(1) + size),
    fill: color,
    stroke: black + 0.3pt,
  )
}

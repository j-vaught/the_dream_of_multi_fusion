#import "@preview/cetz:0.4.2"

#let data = json(sys.inputs.at("data"))
#let canvas-width = 960
#let canvas-height = 540

#set page(
  width: canvas-width * 1pt,
  height: canvas-height * 1pt,
  margin: 0pt,
  fill: rgb(data.background),
)

#cetz.canvas(length: 1pt, {
  import cetz.draw: *

  rect(
    (0, 0),
    (canvas-width, canvas-height),
    fill: rgb(data.background),
    stroke: none,
  )

  for item in data.images {
    content(
      (item.x, canvas-height - item.y - item.height),
      (item.x + item.width, canvas-height - item.y),
      image(
        item.path,
        width: item.width * 1pt,
        height: item.height * 1pt,
        fit: "stretch",
      ),
      padding: 0,
    )
  }

  for item in data.rectangles {
    let item-fill = if item.fill == none {
      none
    } else {
      rgb(item.fill).transparentize(item.alpha * 100%)
    }
    rect(
      (item.x1, canvas-height - item.y2),
      (item.x2, canvas-height - item.y1),
      radius: 0,
      fill: item-fill,
      stroke: (
        paint: rgb(item.stroke),
        thickness: item.width * 1pt,
        dash: item.dash,
      ),
    )
  }

  for item in data.lines {
    line(
      (item.x1, canvas-height - item.y1),
      (item.x2, canvas-height - item.y2),
      stroke: (
        paint: rgb(item.stroke),
        thickness: item.width * 1pt,
        dash: item.dash,
      ),
    )
  }

  for item in data.circles {
    let item-fill = if item.fill == none {
      none
    } else {
      rgb(item.fill).transparentize(item.alpha * 100%)
    }
    circle(
      (item.cx, canvas-height - item.cy),
      radius: item.radius,
      fill: item-fill,
      stroke: (
        paint: rgb(item.stroke),
        thickness: item.width * 1pt,
      ),
    )
  }
})

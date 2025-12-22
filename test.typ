#set page(width: 10cm, height: auto)
#set heading(numbering: "1.")
#let t1 = $n-1$
#let t2 = $n-2$
= Fibonacci  sequence
The Fibonacci sequence is defined through the recurrence relation $F_n = F_t1 + F_t2$.
It can also be expressed in _closed form:_

$ F_n = round(1 / sqrt(5) phi.alt^n), quad
phi.alt = (1 + sqrt(5)) / 2 $

#let count = 8
#let nums = range(1, count +1)
#let fib(n) = (
    if n <= 2 {1}
    else { fib(n - 1) + fib(n - 2)}
)

The first #count numbers of the sequence are:

#align(center, table(
    columns: count,
    ..nums.map(n => $F_#n$),
    ..nums.map(n => str(fib(n))),
))

#set table(
  stroke: none,
  gutter: 0.2em,
  fill: (x, y) =>
    if x == 0 or y == 0 { gray },
  inset: (right: 1.5em),
)

#show table.cell: it => {
  if it.x == 0 or it.y == 0 {
    set text(white)
    strong(it)
  } else if it.body == [] {
    // Replace empty cells with 'N/A'
    pad(..it.inset)[_N/A_]
  } else {
    it
  }
}

#let a = table.cell(
  fill: green.lighten(60%),
)[A]
#let b = table.cell(
  fill: aqua.lighten(60%),
)[B]

#table(
  columns: 4,
  [], [Exam 1], [Exam 2], [Exam 3],

  [John], [], a, [],
  [Mary], [], a, a,
  [Robert], b, a, b,
)
#set document(
  title: [Diversification in Multiple Pairs Trading]
)
#align(center)[#title()]
Implementation based on work from

Ning, B. (2024). _Quantitative Methods of Statistical Arbitrage_.

#outline()




#v(3em)
= Basic Pairs Trading Framework
For each pair,
1. Construct a spread
2. Estimate the hedge ratio (eg. OU approach)
3. Trade on spread


#pagebreak()
#include("pairs_construction.typ")

#pagebreak()
#include("trading.typ")
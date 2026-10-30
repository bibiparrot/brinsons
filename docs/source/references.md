# Sources and conventions

The formulas on this site state the behavior of **this package**. Where a published family has several conventions, the method page gives the exact implemented one.

- Brinson, Hood & Beebower (1986), [*Determinants of Portfolio Performance*](https://doi.org/10.2469/faj.v42.n4.39).
- Brinson & Fachler (1985), [*Measuring Non-U.S. Equity Portfolio Performance*](https://doi.org/10.3905/jpm.1985.409005).
- Cariño (1999), *Combining Attribution Effects Over Time*, *Journal of Performance Measurement* 3(4).
- Groupe de Recherche en Attribution de Performance (1997), *Synthèse des modèles d'attribution de performance*.
- Frongello (2002), *Linking Single Period Attribution Results* and [*Attribution Linking: Proofed and Clarified*](https://frongello.com/support/Works/JPMFall2002.pdf).
- Menchero (2000), *An Optimized Approach to Linking Attribution Effects Over Time*; Menchero (2004), [*Multiperiod Arithmetic Attribution*](https://doi.org/10.2469/faj.v60.n4.2638).
- Karnosky & Singer (1994), [*Global Asset Management and Performance Attribution*](https://rpc.cfainstitute.org/research/foundation/1994/global-asset-management-and-performance-attribution).
- Bacon (2008), *Practical Portfolio Performance Measurement and Attribution*, 2nd ed.; Bacon (2019), [*Performance Attribution: History and Progress*](https://rpc.cfainstitute.org/research/foundation/2019/performance-attribution-history-and-progress).
- Hentschel (2026), *Brinson and Factors: A Unified Framework for Complete Portfolio Return Attribution*; see the [implemented method note](hentschel.md).

The local research notes in `docs/papers` give extended discussion, including distinctions between pybrinson's historical behavior and this implementation. In particular, this package fixes Menchero's equal-cumulative-return edge case and implements GRAP and Frongello with the same factor form. The additive currency split and independent log-premium grid are distinct conventions, each documented on its own page.

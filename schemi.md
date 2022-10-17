# Appunti Calcolo Numerico

## Accuratezza di un risultato

- **Errore assoluto**: $E_a = | R_{approssimato} - R_{esatto} |$
- **Errore relativo**: $E_r = \frac{|E_a|}{|\text{ris esatto}|}$
- **Errore percentuale**: $E_p = (E_r \times 100)\%$

## Rappresentazione dei numeri in memoria

Vengono rappresentati in forma binaria (base 2)

Fissato un numero $\beta \in \N$, ocn $\beta > 1$, rappresentiamo nella base $\beta$ un numero qualunque $\alpha \in \R$

$$\alpha = \pm (\alpha_0 \alpha_1 \dots \alpha_p \text{ . } \alpha_{p+1} \dots)\beta = \sum_{k=0}^{p}{\alpha_k \beta^{p-k}} + \sum_{k=1}^{\infty}{\alpha_{k+p} \beta^{-k}}$$

Normalizzata: $\alpha = \pm(\text{. } \alpha_1 \alpha_2 \dots) \beta^p$

## Numeri Interi
### Rappresentazione in modulo e segno
Il modo più semplice per rappresentarli in memoria è quello di anteporre un bit che vale 0 se è positivo, 1 se negativo.

Con $N$ bit si possono rappresentare tutti i numeri in $[-(2^{N-1} - 1), 2^{N-1} - 1]$

## Numeri Reali

Si usa il metodo delle moltiplicazioni successive

| Moltiplicazione | Parte intera |
| --- | --- |
| $0.2 \times 2 = 0.4$ | $0$ |
| $0.4 \times 2 = 0.8$ | $0$ |
| $0.8 \times 2 = 1.6$ | $1$ |
| $0.6 \times 2 = 1.2$ | $1$ |
| $0.2 \times 2 = 0.4$ | $0$ |

$$(0.2)_{10} = (0.\overline{00110})_2$$

### Rappresentazione scientifica normalizzata di un numero reale

Ogni $x \in \R$ può essere espresso come
$$x = \pm(0 \text{ . } d_1 d_2 \dots)\beta^p = \pm ( \sum_{i=1}^{\infty}{d_i b^{-i}} ) \beta^p \qquad p \in \N, 0 \leq a_i \leq \beta-1, d_1 \neq 0$$

Il numero $0 \text{ . } d_1 d_2 \dots$ viene detto *mantissa* di $x$, $\beta^p$ è la parte esponenete. $p$ è detto *caratteristica* di $x$ (o esponente), $\beta$ è la *base*, $d_i$ sono le *cifre di rappresentazione*

### Sistema floating point

Si definisce un insieme di numeri macchina (*floating-point*) con $t$ cifre significative, base $\beta$ e range $(L, U)$

$$F (\beta, t, L, U) = {0} \cup {x \in R = sign(x) \beta^p \sum_{i=1}^t{d_i \beta^{-1}}} \above{} 
\text{con } t, \beta \in N, \beta \geq 2, 0 \leq d_i \leq \beta - 1, d_1 \neq 0, L \leq p \leq U$$

Nota: $L$ e $U$ sono dello stesso ordine di grandezza e $t$ rappresenta il numero di cifre della matissa (la precisione)

- $F(2, 24, -128, 127)$: precisione singola, 32 bit di cui $24$ per la mantissa e $8$ per l'esponente
- $F(2, 53, -1024, 1023)$: precisione doppia, 64 bit di cui $53$ per la mantissa e $11$ per l'esponente

## Errori di rappresentazione

- Errore assoluto di arrotondamento: $$
- Errore relativo di arrotondamento: $$
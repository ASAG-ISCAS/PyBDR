# Contributing

---

## some tips for coding style

+ The functions shall not try to handle the degenerate cases, rather than they should be handled by
  the caller. rationale being that
    + they slow down computation for the common (non-degenerate) case
    + it's not clear what it means to `catch` these degeneracies

## commonly used abbreviations for coding conventions

| Word        | Abbrev |          Example           | 
|-------------|:------:|:--------------------------:|
| to          |   2    |        cvt2polytope        |
| convert     |  cvt   |        cvt2polytope        |
| constraint  |  cons  | approx_mink_diff_cons_zono | 
| approximate | approx | approx_mink_diff_cons_zono |
| Minkowski   |  mink  | approx_mink_diff_cons_zono |
| zonotope    |  zono  | approx_mink_diff_cons_zono |
    
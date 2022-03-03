# Contributing

---

## some tips for coding style

+ The functions shall not try to handle the degenerate cases, rather than they should be handled by
  the caller. rationale being that
    + they slow down computation for the common (non-degenerate) case
    + it's not clear what it means to `catch` these degeneracies
+ We prefer using API with explicit calling but shorthand without messy too much, for the reason
    + the code is easy to understand
    + can avoid potential bugs introduced by unnecessary typo
        + i.e. we would like to call `power()` explicitly rather than its shorthand `**`
+ The structure of this toolbox is mainly function-based, because there are some algorithms cited
  from published papers may design quite complicated algorithm to realize some kind of `simple` API
    + do minkowski addition/subtraction for some specific geometry objects is quite detailed and
      complicated

## commonly used abbreviations for coding conventions

| Word        | Abbrev |          Example           | 
|-------------|:------:|:--------------------------:|
| to          |   2    |        cvt2polytope        |
| convert     |  cvt   |        cvt2polytope        |
| constraint  |  cons  | approx_mink_diff_cons_zono | 
| approximate | approx | approx_mink_diff_cons_zono |
| Minkowski   |  mink  | approx_mink_diff_cons_zono |
| zonotope    |  zono  | approx_mink_diff_cons_zono |
    
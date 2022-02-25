## Some Tips for Development

+ The functions shall not try to handle the degenerate cases, rather than they should be handled by
  the caller. rationale being that
    + they slow down computation for the common (non-degenerate) case and
    + it's not clear what it means to `catch` the se degeneracies.
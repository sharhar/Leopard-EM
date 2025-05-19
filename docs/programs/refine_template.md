---
title: The Refine Template Program
description: Description of the refine template program and its configuration
---

# The refine template program

Although the match template program samples millions of points across orientation space, this can be considered relatively coarse search compared to the theoretical angular accuracy.
To achieve an even higher angular accuracy, we finely sample SO(3) space locally for each match template identified particle using the `refine_template` program.
A defocus refinement is also included in the `refine_template` program  whose relative sampling is another configurable parameter.
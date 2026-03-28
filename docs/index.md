# LMX Documentation

LMX is a JAX-native solver for inductionless liquid-metal MHD.
The documentation is organized around the solver first, with validation backends
and external comparison assets kept separate.

## Core Guides

```{toctree}
:maxdepth: 2
:caption: Guides

theory
developer_guide
case_cookbook
validation_report
```

## Build and read locally

Build these pages with Read the Docs or a local Sphinx build. The source pages are
Markdown, so the docs stay close to the code and can evolve without changing the
public identity of the project.

## Validation context

External recovered FreeMHD/OpenFOAM cases are optional validation assets. They help
compare LMX against paper cases, but they are not the definition of LMX.

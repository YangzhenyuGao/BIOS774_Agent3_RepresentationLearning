# Sycamore personal R package migration

This is a one-time personal R setup requested by the user, separate from Agent3's
Python analysis environment. The input inventory is `prompt/install.md`.

The original main-list installation is complete. Verification job `5003267` completed with exit
code 0: all 139 requested packages load, all four GitHub commit pins match,
Arrow/HDF5 roundtrips and ImageMagick image creation pass, and a BCalm MPRASet
can be constructed and its counts retrieved. Of the requested packages, 102
are in the personal library and 37 are supplied by the module. The personal
library, including dependencies, occupied approximately 2.2 GB at that stage.
See `logs/r_home_setup/verify-5003267.log` and `verification.tsv` for results.
These are loading and basic functionality checks, not full scientific workflow tests.

- Runtime: `module load r/4.6.0` on Sycamore (RHEL 10.2).
- Destination: `/nas/sycamore/home/gaoyang/R/x86_64-pc-linux-gnu-library/4.6`.
- Scope: the 139 main-list packages and required dependencies. The user also
  requested all 169 Appendix B packages on November 2, 2026. This extension is
  tracked separately in `optional_packages.txt`; 66 were already installed and
  103 were missing at the start. Installation job `5003271` completed successfully
  (exit 0, elapsed 11 minutes 20 seconds). All 169 Appendix B packages and all 139
  main-list packages passed namespace loading; the original commit-pin and smoke
  tests also passed. The full personal library now occupies approximately 3.9 GB.
- Existing module packages are reused; missing packages are built from current
  CRAN/Bioconductor sources. RHEL 9 binaries from the input guide are not used on
  this RHEL 10 host. SeuratData and SeuratDisk retain the exact requested commits.
- The input inventory labels BCalm as Bioconductor, but it is unavailable from
  the current Bioconductor repository. Its author's GitHub installation route is
  used instead: `kircherlab/BCalm@d5164ba4b62107290085de376822fff07c7d40aa`,
  the upstream commit inspected during this migration.
- The gghalves r-universe repository index returned HTTP 404. The author's source
  is used instead: `erocoar/gghalves@e5c3c79e79f13a00795a2a7774794252d1980d86`.
- Module-provided CmdStan 2.39.0 is reused. No personal CmdStan copy is needed.
- Longleaf libraries, system R, existing home startup files, and Agent3's Conda
  environment are not modified by these scripts.

From the Agent3 project root, `sbatch scripts/r_home_setup/install.sbatch` runs
the installation and verification. Do not launch concurrent installers against
the same personal library. The script skips installed main-list packages; if a
package fails verification, inspect the log and repair that package specifically.

Logs, before/after inventories, verification and session information are under
`logs/r_home_setup/`. Compilation scratch is also kept there. A successful job
requires all 139 packages to load, all recorded GitHub pins to match, working Arrow and
HDF5 roundtrips, and basic ImageMagick functionality. No scientific analysis or
Agent3 training is performed.

For BCalm, run `library(BCalm)` before constructing MPRASet objects. Its mpra
dependency resolves `Biobase::validMsg` through attached dependencies; calling
only `BCalm::MPRASet()` in a fresh session failed the smoke test. Normal package
attachment passed without modifying third-party package code.

For subsequent use:

```bash
module load r/4.6.0
Rscript -e '.libPaths()'
R
```

R automatically searches the existing version-specific home library; no
`.Renviron` change is needed for the default module configuration.

To install or resume Appendix B from the project root:

```bash
sbatch scripts/r_home_setup/install_optional.sbatch
```

Do not run this concurrently with another installer targeting the same library.
It skips installed packages and uses `update = FALSE`. The final check loads all
308 main/Appendix B packages and repeats the main-list smoke tests and commit-pin
checks. Results are in `logs/r_home_setup/optional_verification.tsv`, with
before/after inventories and session information prefixed `optional_`.
The successful extension log is `logs/r_home_setup/optional-5003271.log`.
Loading emitted an RSQLite method-registration notice from lumi and Arrow API
deprecation warnings during smoke tests; neither caused a validation failure.
Package-specific scientific analyses have not been exhaustively tested.

"""Optional behavioural / non-genetic extensions of the Manas simulator,
added to score Temura Chinmay Krishna Yadav's 2019 BS-MS thesis (IISER Mohali)
and its contrast papers. Specification: RD_simulator_extensions_chinmay.docx
and acceptance_tests_extensions.csv (mate_choice_chinmay/).

Every module is OFF by default and nothing in the genetic core (``iasc``)
imports this package, so with all flags off the existing Manas validation is
untouched (tests B3 and A5).

  quality   female infection state, genotype-specific clearance, fecundity cost
            kappa, perceived quality cue, decapitation
  choice    softmax courtship kernel; courts-first (CF), courtship latency (CL),
            courts-most (CM), copulation duration; Khan-type group mating assay
  harm      mate harm as a pleiotropic side effect (Morrow et al. 2003 response)
  telegony  compartmentalisation switch for first-male (stepfather) effects
  coupling  read a genetic-layer genotype record (iasc/ibm/regime.py) and
            give kappa, beta and survival per genotype (coupler.py,
            docs/09_coupling.md); off = the global constants of the regime
"""

MODULES = ("quality", "choice", "harm", "telegony", "coupling")
DEFAULT_FLAGS = {m: False for m in MODULES}


class ModuleOff(RuntimeError):
    """Raised when a module's function is called while its flag is off."""


def require(flags, name):
    if not flags.get(name, False):
        raise ModuleOff(f"extension module '{name}' is off (set modules.{name}: true in the regime config)")

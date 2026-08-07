from dataclasses import dataclass
import yaml

@dataclass
class Lattice:
    T: int

@dataclass
class Statistics:
    stati: int
    configuration: int
    bootstrap: int

@dataclass
class Analysis:
    subtraction: int
    normalization: int
    init: int
    size: int
    dof: int
    fitini: int
    fitfin: int
    svdrankmax: int

@dataclass
class Extrapolation:
    rankmin: int
    rankmax: int
    msize: int

@dataclass
class Mock:
    state: int
    error: float

@dataclass
class Paths:
    corr: str
    diag: str

@dataclass
class Files:
    rdata: str
    cdata: str
    ndata: str

@dataclass
class Config:
    lattice: Lattice
    statistics: Statistics
    analysis: Analysis
    extrapolation: Extrapolation
    mock: Mock
    paths: Paths
    files: Files

    @classmethod
    def from_yaml(cls, filename):

        with open(filename) as f:
            raw = yaml.safe_load(f)

        return cls(
            lattice=Lattice(**raw["lattice"]),
            statistics=Statistics(**raw["statistics"]),
            analysis=Analysis(**raw["analysis"]),
            extrapolation=Extrapolation(**raw["extrapolation"]),
            mock=Mock(**raw["mock"]),
            paths=Paths(**raw["paths"]),
            files=Files(**raw["files"]),
        )

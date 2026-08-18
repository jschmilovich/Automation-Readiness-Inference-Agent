# Automation Readiness Inference Agent
"""Load the scored workflow set into clean records.

Each row of workflows.csv is one workflow with SaxeCap's labels attached.
Those labels (utility, complexity, roi) are your benchmark answer key.
"""
from dataclasses import dataclass
import pandas as pd


@dataclass
class Workflow:
    wid: int                # row id, used as the unique handle
    department: str
    name: str
    description: str
    utility: str            # SaxeCap label (may be blank)
    complexity: str         # SaxeCap label (may be blank)
    roi: str                # SaxeCap label (may be blank)
    source: str

    @property
    def text(self) -> str:
        # what retrieval and scoring actually read
        parts = [self.name, self.department, self.description]
        return " . ".join(p for p in parts if p)


def load_corpus(path: str = "workflows.csv") -> list[Workflow]:
    df = pd.read_csv(path).fillna("")
    rows = []
    for i, r in df.iterrows():
        rows.append(Workflow(
            wid=int(i),
            department=str(r.get("department", "")),
            name=str(r.get("initiative", "")),
            description=str(r.get("description", "")),
            utility=str(r.get("utility_or_impact", "")),
            complexity=str(r.get("complexity", "")),
            roi=str(r.get("roi", "")),
            source=str(r.get("source_doc", "")),
        ))
    return rows


if __name__ == "__main__":
    wfs = load_corpus()
    print(f"loaded {len(wfs)} workflows")
    for w in wfs[:5]:
        print(f"  [{w.wid}] {w.name}  ({w.department})  util={w.utility} cx={w.complexity}")

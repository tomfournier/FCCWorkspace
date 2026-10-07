from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import ROOT



def setup_alias(df: 'ROOT.ROOt.RDataFrame',
                cat: str
                ) -> 'ROOT.ROOT.RDataFrame':
    """Attach collection aliases based on final state.

    Args:
        df: RDataFrame-like object used throughout the selection chain.
        cat (str): Final-state category, either 'mumu' or 'ee'.

    Returns:
        The input dataframe with lepton, MC association, and particle aliases defined.

    Raises:
        ValueError: If an unsupported category is provided.
    """
    # Alias for MC truth matching and particle collections
    df = df.Alias('MCRecoAssociations0', 'MCRecoAssociations#0.index')
    df = df.Alias('MCRecoAssociations1', 'MCRecoAssociations#1.index')
    df = df.Alias('Particle0', 'Particle#0.index')
    df = df.Alias('Particle1', 'Particle#1.index')
    df = df.Alias('Photon0',   'Photon#0.index')

    # Alias for lepton collections based on final state
    if cat == 'mumu':
        df = df.Alias('Lepton0', 'Muon#0.index')
    elif cat == 'ee':
        df = df.Alias('Lepton0', 'Electron#0.index')
    elif cat == 'qq':
        df = df.Alias('Muon0',     'Muon#0.index')
        df = df.Alias('Electron0', 'Electron#0.index')
    else:
        raise ValueError(f'cat {cat} is not supported, choose between [ee, mumu, qq]')

    return df


def cutflow(df: 'ROOT.ROOT.RDataFrame',
            hists: list['ROOT.TH1'],
            cut: int
            ) -> 'ROOT.ROOT.RDataFrame':

    # Ensure no duplicate column is defined
    if f'cut{cut}' in df.GetColumnNames():
        raise ValueError(f'cut{cut} is already defined')

    df = df.Define(f'cut{cut}', str(cut))
    hists.append(df.Histo1D(('cutFlow', '', 50, 0, 50), f'cut{cut}'))
    return df, hists

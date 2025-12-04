#!/usr/bin/env python3
"""
PPPx INI Configuration Generator

Generates dynamic pppx.ini files for positioning evaluation with:
- Dynamic year/DOY parameters
- Configurable ionosphere sources (custom STEC CSV or IGS GIM)
- Experiment-specific output paths
"""

import os
from pathlib import Path
from datetime import datetime


def generate_pppx_ini(
    year,
    doy,
    output_path,
    products_dir,
    ion_source="IONEX",
    ion_path=None,
    station_name=None,
    mode="kinematic",
    output_dir="./",
    elev_mask=7,
    use_relative_paths=True,
    output_ini_dir=None
):
    """
    Generate pppx.ini configuration file.
    
    Args:
        year: Year (int)
        doy: Day of year (int)
        output_path: Path to save INI file
        products_dir: Directory containing products (SP3, CLK, etc.)
        ion_source: Ionosphere source - "IONEX" for GIM, "CSV" for custom STEC
        ion_path: Path to ionosphere file (IONEX or CSV)
        station_name: Station name (for output filename)
        mode: Position mode - "kinematic" or "static"
        output_dir: Output directory for positioning results
        elev_mask: Elevation mask in degrees
        use_relative_paths: Use relative paths in INI (for short paths)
        output_ini_dir: Directory where INI will be placed (for computing relative paths)
    
    Returns:
        Path to generated INI file
    """
    # Compute relative paths from where the INI will be located
    if output_ini_dir:
        ini_dir = Path(output_ini_dir)
    else:
        ini_dir = Path(output_path).parent
    
    # Make product paths relative to INI location
    products_path = Path(products_dir)
    if use_relative_paths:
        try:
            products_rel = os.path.relpath(products_path, ini_dir)
            products_path = Path(products_rel)
        except ValueError:
            pass  # Keep absolute if relpath fails
    
    # Construct product filenames
    sp3_file = products_path / f"COD0OPSFIN_{year}{doy:03d}0000_01D_05M_ORB.SP3"
    clk_file = products_path / f"COD0OPSFIN_{year}{doy:03d}0000_01D_30S_CLK.CLK"
    erp_file = products_path / f"COD0OPSFIN_{year}{doy:03d}0000_01D_01D_ERP.ERP"
    obx_file = products_path / f"COD0OPSFIN_{year}{doy:03d}0000_01D_30S_ATT.OBX"
    
    # Determine ionosphere file
    # PPPx uses iono_model = "IONEX" for both IONEX and CSV files
    # It auto-detects the format based on file extension (.INX vs .csv)
    iono_model = "IONEX"
    
    if ion_path:
        ion_p = Path(ion_path)
        if use_relative_paths:
            try:
                iono_file = Path(os.path.relpath(ion_p, ini_dir))
            except ValueError:
                iono_file = ion_p
        else:
            iono_file = ion_p
    else:
        # Default to CODE GIM if no path provided
        iono_file = products_path / f"COD0OPSFIN_{year}{doy:03d}0000_01D_01H_GIM.INX"
    
    # Get table directory - relative to INI location
    pppx_dir = Path(__file__).parent
    if use_relative_paths:
        try:
            table_rel = os.path.relpath(pppx_dir / "table", ini_dir)
            table_dir = Path(table_rel)
        except ValueError:
            table_dir = pppx_dir / "table"
    else:
        table_dir = pppx_dir / "table"
    
    # Use relative path for output
    output_dir_path = Path(output_dir)
    
    # Generate INI content
    ini_content = f"""; PPPx configuration file - Auto-generated
; Date: {year}/{doy:03d}
; Ionosphere: {ion_source}
; Station: {station_name if station_name else 'N/A'}

[session]
interval =                      ; opt: 0:RINEX-OBS default [sec]
date     = {year} {doy}         ; year DOY
start    =                      ; hour min sec
end      =                      ; hour min sec

[constellation]
system  = GE                    ; opt: GRECJ    [GRECJ]
exclude =                       ; PRNs to be excluded

[observation]
noise = 0.3 0.002               ; observation noise of code/phase (m) [ 0.3 0.002 ]
; G/R/E/C/J = f_1 f_2 obs_priority: high -> low
G = 1 2 PWCLSXYMN               ; default: [ 1 2 PWCLSXYMN ]
R = 1 2 PCIQX                   ; default: [ 1 2 PCIQX ]
E = 1 5 BCIQX                   ; default: [ 1 5 BCIQX ]
C = 2 6 IQX                     ; default: [ 2 6 IQX ]
J = 1 2 SLXCZ                   ; default: [ 1 2 SLXCZ ]

[model]
trop = GMF                      ; opt: GMF/VMF1/GPT2w/none      [ GMF ]
iono = {iono_model}             ; opt: IF/brdc/IONEX/CSV/none   [ IF ]

[solver]
sol_mode   = ppp                ; opt: spp/ppp/rtk/tdp          [ spp ]
pos_mode   = {mode}             ; opt: kinematic/static/fixed   [ kinematic ]
solver     = fgo                ; opt: fgo/kalman/lsq           [ kalman ]
weight_opt = elev               ; opt: elev/snr                 [ elev ]
elev_mask  = {elev_mask}        ; elevation mask (°)            [ 10 ]
snr_mask   = 25                 ; SNR mask (dB-Hz)              [ 35 ]
slip_det   = ALL                ; opt: off GF MW LLI ALL        [ all ]
pos_pri    = 100 1              ; uncertainty  process_noise    [ 1E+03 1 ]
clk_pri    = 100 100            ;      m         m/sqrt(s)      [ 5E+03 3E+03 ]
isb_pri    = 50  3.2E-04        ;                               [ 5E+03 3.2E-04 ]
ztd_pri    = 0.5 3E-05          ;                               [ 0.5 1E-05 ]
sf_ppp     = yes                ; yes/no: SF-PPP (only for ppp) [ no ]

[product]
src = precise                   ; opt: brdc/precise
sp3 = {sp3_file}                ; precise orbit
clk = {clk_file}                ; precise clock
erp = {erp_file}                ; earth rotation parameters
obx = {obx_file}                ; satellite attitude
ion = {iono_file}               ; ionosphere model (IONEX or CSV)

[table]
igsatx    = {table_dir}/igs20.atx
oceanload = {table_dir}/oceanload
channel   = {table_dir}/glonass_chn
gpt2w     = {table_dir}/gpt2_1wA.grd
orography = {table_dir}/orography_ell

[output]
path  = {output_dir_path}
level = info                    ; opt: off/critical/error/warn/info/debug/trace
"""
    
    # Write to file
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        f.write(ini_content)
    
    return output_file


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate pppx.ini configuration")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--doy", type=int, required=True)
    parser.add_argument("--output", type=str, required=True, help="Output INI path")
    parser.add_argument("--products_dir", type=str, required=True)
    parser.add_argument("--ion_source", type=str, default="IONEX", choices=["IONEX", "CSV"])
    parser.add_argument("--ion_path", type=str, default=None)
    parser.add_argument("--station", type=str, default=None)
    
    args = parser.parse_args()
    
    ini_path = generate_pppx_ini(
        args.year,
        args.doy,
        args.output,
        args.products_dir,
        ion_source=args.ion_source,
        ion_path=args.ion_path,
        station_name=args.station
    )
    
    print(f"Generated: {ini_path}")

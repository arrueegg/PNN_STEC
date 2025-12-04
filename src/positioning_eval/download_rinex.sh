#!/bin/bash

###############################################################################
##  RINEX Download Script for CDDIS Archive                                 ##
##  Downloads observation files for specified stations and date             ##
###############################################################################

download_rinex() {
    # Usage: download_rinex STATION YEAR DOY OUTPUT_DIR
    local station="$1"
    local year="$2"
    local doy="$3"
    local output_dir="$4"
    
    local station_upper="${station^^}"
    local station_lower="${station,,}"
    local yy="${year:2:2}"
    
    mkdir -p "$output_dir"
    
    # Try RINEX 3 long filename format first
    local long_filename="${station_upper}00CHE_R_${year}${doy}0000_01D_30S_MO.crx.gz"
    local long_url="https://cddis.nasa.gov/archive/gnss/data/daily/${year}/${doy}/${yy}d/${long_filename}"
    
    # Try RINEX 2 short filename format
    local short_filename="${station_lower}${doy}0.${yy}d.Z"
    local short_url="https://cddis.nasa.gov/archive/gnss/data/daily/${year}/${doy}/${yy}d/${short_filename}"
    
    cd "$output_dir" || return 1
    
    # Create empty cookies file if it doesn't exist
    touch cookies.txt
    
    # Clean up any previous failed downloads (HTML error pages)
    [ -f "$long_filename" ] && ! file "$long_filename" | grep -q "gzip compressed" && rm -f "$long_filename"
    [ -f "$short_filename" ] && ! file "$short_filename" | grep -q "compress'd data" && rm -f "$short_filename"
    
    # Try long format (use ~/.netrc for authentication with NASA Earthdata)
    # Need to handle cookies and redirects for CDDIS authentication
    if wget --netrc --auth-no-challenge=on --keep-session-cookies --save-cookies=cookies.txt --load-cookies=cookies.txt -nv -nc -c -t 3 --connect-timeout=10 --read-timeout=60 "$long_url" 2>&1; then
        if [ -f "$long_filename" ]; then
            # Check if it's actually gzipped (not HTML error page)
            if file "$long_filename" | grep -q "gzip compressed"; then
                gunzip -f "$long_filename" 2>/dev/null
                local crx_file="${long_filename%.gz}"
                if [ -f "$crx_file" ]; then
                    # Convert Hatanaka to RINEX
                    local converter=""
                    if command -v CRX2RNX &> /dev/null; then
                        converter="CRX2RNX"
                    elif command -v crx2rnx &> /dev/null; then
                        converter="crx2rnx"
                    elif [ -f "$HOME/.local/bin/crx2rnx" ]; then
                        converter="$HOME/.local/bin/crx2rnx"
                    fi
                    
                    if [ -n "$converter" ]; then
                        $converter "$crx_file" 2>/dev/null
                        rm -f "$crx_file"
                        echo "Downloaded and converted: ${station_upper}"
                        return 0
                    else
                        # CRX2RNX not available - keep .crx file (PPPx can handle it)
                        echo "Downloaded: ${station_upper} (Hatanaka .crx format)"
                        return 0
                    fi
                fi
            else
                rm -f "$long_filename"
            fi
        fi
    fi
    
    # Try short format (use ~/.netrc for authentication with NASA Earthdata)
    if wget --netrc --auth-no-challenge=on --keep-session-cookies --load-cookies=cookies.txt -nv -nc -c -t 3 --connect-timeout=10 --read-timeout=60 "$short_url" 2>&1; then
        if [ -f "$short_filename" ]; then
            # Check if it's actually compressed (not HTML)
            if file "$short_filename" | grep -q "compress'd data"; then
                uncompress "$short_filename" 2>/dev/null
                echo "Downloaded: ${station_upper}"
                return 0
            else
                rm -f "$short_filename"
            fi
        fi
    fi
    
    echo "Failed to download RINEX for ${station_upper}"
    return 1
}

# Main entry point
if [ $# -ge 3 ]; then
    download_rinex "$1" "$2" "$3" "${4:-.}"
else
    echo "Usage: download_rinex.sh STATION YEAR DOY [OUTPUT_DIR]"
    echo "Example: download_rinex.sh ZIMM 2024 183 ./rinex"
    exit 1
fi

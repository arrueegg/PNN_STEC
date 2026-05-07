#!/bin/bash

###############################################################################
##  RINEX Download Script for CDDIS Archive                                 ##
##  Downloads observation files for specified stations and date             ##
###############################################################################

# Cache directory listing to avoid repeated requests
if [ -n "$RINEX_CACHE_DIR" ]; then
    CACHE_DIR="$RINEX_CACHE_DIR"
else
    CACHE_DIR="/tmp/rinex_cache_$$"
    trap 'rm -rf "$CACHE_DIR"' EXIT
fi
mkdir -p "$CACHE_DIR"

get_directory_listing() {
    local year="$1"
    local doy="$2"
    local yy="${year:2:2}"
    local base_url="https://cddis.nasa.gov/archive/gnss/data/daily/${year}/${doy}/${yy}d"
    local cache_file="$CACHE_DIR/listing_${year}_${doy}.txt"
    
    # Return cached listing if it exists
    if [ -f "$cache_file" ]; then
        cat "$cache_file"
        return 0
    fi
    
    # Fetch and cache directory listing
    wget --netrc --auth-no-challenge -q -O - "${base_url}/" 2>/dev/null | tee "$cache_file"
}

download_rinex() {
    # Usage: download_rinex STATION YEAR DOY OUTPUT_DIR [KNOWN_FILENAME]
    local station="$1"
    local year="$2"
    local doy="$3"
    local output_dir="$4"
    local known_filename="$5"  # Optional: if provided, use this exact filename
    
    local station_upper="${station^^}"
    local station_lower="${station,,}"
    local yy="${year:2:2}"
    
    mkdir -p "$output_dir"
    
    # Try RINEX 3 long filename format - use wildcard to match any country code
    # Format: SSSS00CCC_R_YYYYDDDHHMM_01D_30S_MO.crx.gz
    # where SSSS=station, CCC=3-letter country code
    local long_pattern="${station_upper}00???_R_${year}${doy}0000_01D_30S_MO.crx.gz"
    local base_url="https://cddis.nasa.gov/archive/gnss/data/daily/${year}/${doy}/${yy}d"
    
    # Try RINEX 2 short filename format
    local short_filename="${station_lower}${doy}0.${yy}d.Z"
    local short_url="https://cddis.nasa.gov/archive/gnss/data/daily/${year}/${doy}/${yy}d/${short_filename}"
    
    cd "$output_dir" || return 1
    
    # Create empty cookies file if it doesn't exist
    touch cookies.txt
    
    # Get directory listing (cached) and find the matching file for this station
    local dir_listing=$(get_directory_listing "$year" "$doy")
    
    # Extract the RINEX 3 filename for this station (any country code)
    local long_filename=$(echo "$dir_listing" | grep -oE "${station_upper}00[A-Z]{3}_R_${year}${doy}0000_01D_30S_MO\.crx\.gz" | head -1)
    
    if [ -n "$long_filename" ]; then
        # Found the file - download it
        local long_url="${base_url}/${long_filename}"
        
        # Download with authentication and retries
        local max_attempts=5
        local attempt=1
        local delay=5
        local success=0
        
        while [ $attempt -le $max_attempts ]; do
            # Clean up any previous failed attempts
            [ -f "$long_filename" ] && ! file "$long_filename" | grep -q "gzip compressed" && rm -f "$long_filename"
            
            if wget --netrc --auth-no-challenge=on --keep-session-cookies --save-cookies=cookies.txt --load-cookies=cookies.txt -nv -nc -c -t 3 --connect-timeout=10 --read-timeout=60 "$long_url" 2>&1; then
                if [ -f "$long_filename" ] && file "$long_filename" | grep -q "gzip compressed"; then
                    success=1
                    break
                else
                    [ -f "$long_filename" ] && rm -f "$long_filename"
                fi
            fi
            
            echo "Attempt $attempt of $max_attempts failed for $long_filename. Retrying in ${delay}s..."
            sleep $delay
            attempt=$((attempt + 1))
            delay=$((delay * 2))
        done

        if [ $success -eq 1 ]; then
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
                    echo "Downloaded: ${station_upper} (Hatanaka .crx format)"
                    return 0
                fi
            fi
        fi
    fi
    
    # Try short format (use ~/.netrc for authentication with NASA Earthdata)
    local max_attempts=5
    local attempt=1
    local delay=5
    local success=0

    while [ $attempt -le $max_attempts ]; do
        # Clean up any previous failed short format downloads
        [ -f "$short_filename" ] && ! file "$short_filename" | grep -q "compress'd data" && rm -f "$short_filename"

        if wget --netrc --auth-no-challenge=on --keep-session-cookies --load-cookies=cookies.txt -nv -nc -c -t 3 --connect-timeout=10 --read-timeout=60 "$short_url" 2>&1; then
            if [ -f "$short_filename" ] && file "$short_filename" | grep -q "compress'd data"; then
                success=1
                break
            else
                rm -f "$short_filename"
            fi
        fi
        
        echo "Attempt $attempt of $max_attempts failed for $short_filename. Retrying in ${delay}s..."
        sleep $delay
        attempt=$((attempt + 1))
        delay=$((delay * 2))
    done

    if [ $success -eq 1 ]; then
        uncompress "$short_filename" 2>/dev/null
        echo "Downloaded: ${station_upper}"
        return 0
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

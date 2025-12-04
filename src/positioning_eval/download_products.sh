#!/bin/bash


###############################################################################
##                                                                           ##
##  PURPOSE: GNSS data processing with PPPx                                  ##
##                                                                           ##
##  AUTHOR : Yuanxin Pan (yxpan.im@gmail.com)                                ##
##                                                                           ##
##  VERSION: 1.0.0                                                           ##
##                                                                           ##
##    Copyright (C) 2025 by Yuanxin Pan                                      ##
##                                                                           ##
##    This program is free software: you can redistribute it and/or modify   ##
##    it under the terms of the GNU General Public License (version 3) as    ##
##    published by the Free Software Foundation.                             ##
##                                                                           ##
##    This program is distributed in the hope that it will be useful,        ##
##    but WITHOUT ANY WARRANTY; without even the implied warranty of         ##
##    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the          ##
##    GNU General Public License (version 3) for more details.               ##
##                                                                           ##
##    You should have received a copy of the GNU General Public License      ##
##    along with this program.  If not, see <https://www.gnu.org/licenses/>. ##
##                                                                           ##
###############################################################################

main()
{
    [ $# -ne 2 ] && echo "usage: download_products.sh YEAR DOY" && return 1
    local ymd="`ydoy2ymd $1 $2`"
    local mjd=`ymd2mjd $ymd`

    # Download products
    PrepareProducts $mjd "./products" || return 1
}

GetProductNames() { # purpose: Get products name of a specific AC
                    # usage  : GetProductNames mjd ac
    local mjd=$1
    local ac=$2

    local ydoy=($(mjd2ydoy $mjd))
    local wkdow=($(mjd2wkdow $mjd))
    local year=${ydoy[0]}
    local doy=${ydoy[1]}
    local week=${wkdow[0]}
    local dow=${wkdow[1]}

    local sp3="${ac^^}0OPSFIN_${year}${doy}0000_01D_05M_ORB.SP3.gz"
    local clk="${ac^^}0OPSFIN_${year}${doy}0000_01D_30S_CLK.CLK.gz"
    local erp="${ac^^}0OPSFIN_${year}${doy}0000_01D_01D_ERP.ERP.gz"
    local obx="${ac^^}0OPSFIN_${year}${doy}0000_01D_30S_ATT.OBX.gz"
    local bia="${ac^^}0OPSFIN_${year}${doy}0000_01D_01D_OSB.BIA.gz"
    local ion="${ac^^}0OPSFIN_${year}${doy}0000_01D_01H_GIM.INX.gz"

    if [ $mjd -lt 59910 ]; then
        sp3="${ac^^}${week}${dow}.EPH.Z"
        clk="${ac^^}${week}${dow}.CLK.Z"
        erp="${ac^^}${week}${dow}.ERP.Z"
        obx="${ac^^}${week}${dow}.OBX.Z"
        bia="${ac^^}${week}${dow}.BIA.Z"
        ion="${ac^^}G${doy}0.${year:2:2}I.Z"
    fi

    echo "sp3=$sp3; clk=$clk; erp=$erp; obx=$obx; bia=$bia; ion=$ion"
    return 0
}

PrepareProducts() { # purpose: prepare products in working directory
                    # usage  : PrepareProducts mjd products_dir
    local mjd_mid=$1
    local products_dir="$2"

    [ -d $products_dir ] || mkdir -p "$products_dir"

    local product_src="precise"
    local ydoy=($(mjd2ydoy $mjd_mid))
    local wkdow=($(mjd2wkdow $mjd_mid))
    local year=${ydoy[0]}
    local doy=${ydoy[1]}
    local week=${wkdow[0]}
    local dow=${wkdow[1]}

    local ac="COD"
    local HOST="ftp://ftp.aiub.unibe.ch/CODE/$year"
    local sp3 clk erp obx bia ion
    eval $(GetProductNames $mjd_mid $ac)

    # GIM
    local ion_no_suffix=${ion%.*}
    DownloadProduct ${products_dir}/${ion_no_suffix} $HOST/$ion || return 1

    # BRDC
    # local BRDC_HOST="ftp://gssc.esa.int/gnss/data/daily/${year}/brdc"
    # local brdc=$(GetBrdcName $mjd_mid)
    # if [ "$product_src" = "brdc" -o "$ion_opt" = "brdc" ]; then
    #     local brdc_no_suffix=${brdc%.*}
    #     local nav_args="--nav ${products_dir}/${brdc_no_suffix}"
    #     DownloadProduct ${products_dir}/${brdc_no_suffix} $BRDC_HOST/$brdc || return 1
    #     [ "$ion_opt" = "brdc" ] && product_args="$nav_args "
    #     [ "$product_src" = "brdc" ] && product_args="$nav_args $vmf_args $ion_args" && return 0
    # elif [ "$product_src" != "precise" ]; then
    #     echo -e "$ctrl_file: [product] src: not set correctly"
    #     return 1
    # fi

    # Download
    local product_lists="$sp3 $clk $erp $obx"
    for f in $product_lists
    do
        f_no_suffix=${f%.*}
        DownloadProduct ${products_dir}/${f_no_suffix} $HOST/$f
    done
    [ ! -f $products_dir/${sp3%.*} ] && echo -e "Download $sp3 failed" && return 1

    return 0
}

DownloadProduct() { # purpose: download and uncompress a product
                    # usage  : DownloadProduct file_no_suffix url
    local file="$1"
    local url="$2"
    local base_no_suffix=$(basename -- "$url" ".${url##*.}")
    local suffix=".${url##*.}"
    if [ -f "$file" ]; then
        return 0
    else
        WgetDownload "$url" || return 1
        if [ $suffix = '.Z' -o $suffix = '.gz' ]; then
            gunzip -f $(basename "$url") || return 1
        else
            base_no_suffix=$(basename "$url")
        fi
        [ ! "$base_no_suffix" = "$file" ] && mv "$base_no_suffix" "$file"
        return 0
    fi
}

WgetDownload() { # purpose: download a file with wget
                 # usage  : WgetDownload url
    local url="$1"
    local filename=$(basename "${url}")
    
    # Initialize cookie file if it doesn't exist
    [ ! -f cookies.txt ] && touch cookies.txt
    
    local args="--netrc --auth-no-challenge=on --keep-session-cookies --save-cookies=cookies.txt --load-cookies=cookies.txt -nv -nc -c -t 3 --connect-timeout=10 --read-timeout=60"

    # Remove existing file if it's not a valid gzip/compress file (e.g., HTML error page)
    if [ -f "$filename" ] && ! file "$filename" | grep -qE "(gzip compressed|compress'd|ASCII text)"; then
        echo "Removing invalid file: $filename"
        rm -f "$filename"
    fi

    wget ${args} ${url}
    
    # Validate downloaded file
    if [ -e "$filename" ]; then
        # Check if it's actually the correct file type (not an HTML error page)
        if file "$filename" | grep -qE "(gzip compressed|compress'd|ASCII text)"; then
            return 0
        else
            echo "Downloaded file is invalid (possibly HTML error page)"
            rm -f "$filename"
            return 1
        fi
    else
        return 1
    fi
}

ymd2mjd() {
    local year=$1
    local mon=$((10#$2))
    local day=$((10#$3))
    [ $year -lt 100 ] && year=$((year+2000))
    if [ $mon -le 2 ];then
        mon=$(($mon+12))
        year=$(($year-1))
    fi
    local mjd=`echo $year | awk '{print $1*365.25-$1*365.25%1-679006}'`
    mjd=`echo $mjd $year $mon $day | awk '{print $1+int(30.6001*($3+1))+2-int($2/100)+int($2/400)+$4}'`
    #local mjd=$(bc <<< "$year*365.25 - $year*365.25 % 1 - 679006")
    #mjd=$(bc <<< "($mjd + (30.6001*($mon+1))/1 + 2 - $year/100 + $year/400 + $day)/1")
    echo $mjd
}

ydoy2ymd() {
    local iyear=$1
    local idoy=$((10#$2))
    local days_in_month=(31 28 31 30 31 30 31 31 30 31 30 31)
    local iday=0
    [ $iyear -lt 100 ] && iyear=$((iyear+2000))
    local tmp1=$(($iyear%4))
    local tmp2=$(($iyear%100))
    local tmp3=$(($iyear%400))
    if [ $tmp1 -eq 0 -a $tmp2 -ne 0 ] || [ $tmp3 -eq 0 ]; then
       days_in_month[1]=29
    fi
    local id=$idoy
    local imon=0
    local days
    for days in ${days_in_month[*]}
    do
        id=$(($id-$days))
        imon=$(($imon+1))
        if [ $id -gt 0 ]; then
            continue
        fi
        iday=$(($id + $days))
        break
    done
    printf "%d %02d %02d\n" $iyear $imon $iday
}

mjd2ydoy() {
    local mjd=$1
    local year=$((($mjd + 678940)/365))
    local mjd0=$(ymd2mjd $year 1 1)
    local doy=$(($mjd-$mjd0))
    while [ $doy -le 0 ];do
        year=$(($year-1))
        mjd0=$(ymd2mjd $year 1 1)
        doy=$(($mjd-$mjd0+1))
    done
    printf "%d %03d\n" $year $doy
}

mjd2wkdow() {
    local mjd=$1
    local mjd0=44243
    local difmjd=$(($mjd-$mjd0-1))
    local week=$(($difmjd/7))
    local dow=$(($difmjd%7))
    echo $week $dow
}

######################################################################
##                               Entry                              ##
######################################################################
main "$@"

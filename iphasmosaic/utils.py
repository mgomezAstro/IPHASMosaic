import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
import astroalign as aa
import logging
from typing import Optional, Callable, Any, Union
import os
import sys
from pathlib import Path
from tqdm import tqdm


FORMAT = "%(name)s - %(levelname)s:%(message)s"
LOGLEVEL = logging.INFO
logging.basicConfig(level=LOGLEVEL, format=FORMAT)
logger = logging.getLogger(__name__)


def subtract(fits_1: str, fits_2, remove_bkg: bool = False) -> None:
    """Subtracts band1 images from band2 images."""

    hdu1 = fits.open(fits_1)
    hdu2 = fits.open(fits_2)

    # Normalize
    if remove_bkg:
        _, median, _ = sigma_clipped_stats(data=hdu1[1].data, sigma=2.8)
        hdu1[1].data = hdu1[1].data - median
        _, median, _ = sigma_clipped_stats(data=hdu2[1].data, sigma=2.8)
        hdu2[1].data = hdu2[1].data - median

    error_flag = False
    try:
        image_aligned2, _ = aa.register(hdu2[1].data, hdu1[1].data, detection_sigma=2.8)
    except aa.MaxIterError as e:
        error_flag = True
        logger.error(f"Images {fits_1} and {fits_2} could not be aligned. Error: {e}")

    if not error_flag:
        res = hdu1[1].data - image_aligned2
        head = hdu1[1].header

        reshdu = fits.PrimaryHDU(data=res, header=head)
        reshdul = fits.HDUList(reshdu)
        output_name = f"sub_{fits_1.strip('.fits.fz.')}_{fits_2.strip('.fits.fz')}.fits"
        reshdul.writeto(str(output_name), overwrite=True)


def clean_image(image: str, ccd: Optional[int] = 0) -> None:
    """Remove bad pixels using CCD masking."""
    if ccd == 0:
        ccd = image.split("-")[-1][0]
    masks_path = Path(
        os.path.join(os.path.dirname(sys._getframe(1).f_code.co_filename), "masks/")
    )
    mask_file = masks_path / f"mask_{ccd}.fits"
    mask = fits.getdata(str(mask_file))
    mask = mask.astype(bool)

    with fits.open(image) as hdu:
        hdu[0].data = hdu[0].data.astype(np.float32)
        hdu[0].data[mask] = np.nan
        oimage = image.split("/")[-1]
        hdu.writeto(
            str(Path(image).parent / f"clean_{oimage}"),
            overwrite=True,
        )


def unpack(input_file: str, output_file: Optional[str] = None) -> None:
    """Read the .fz file and save it to .fits file."""
    with fits.open(input_file) as hdu:

        if output_file is None:
            output_file = input_file.strip(".fz")

        data = hdu[1].data
        header = hdu[1].header


        hdunew = fits.PrimaryHDU(data, header)
        hdulnew = fits.HDUList([hdunew])
        hdulnew.writeto(output_file, overwrite=True)


def download_frames_by_5(
    tab: pd.DataFrame,
    fnc: Callable[[pd.DataFrame, str], None],
    download_dir: Optional[str] = None,
) -> None:
    m, n, size = 0, 5, len(tab)
    while n < size:
        fnc(tab[m:n], download_dir)
        m += 5
        n += 5
        if n >= size > m:
            n = size - 1
            fnc(tab[m::], download_dir)


def orginize_images(
    tab: pd.DataFrame,
    band: str,
    download_dir: Union[Path, str],
    raw_dir: Union[Path, str],
) -> None:
    im_band = tab.loc[tab["band"] == band].copy()

    for i in tqdm(range(len(im_band)), desc="funpack/cleaning images..."):
        file = f'r{im_band.iloc[i]["run"]}-{im_band.iloc[i]["ccd"]}.fits.fz'
        ofile = f'r{im_band.iloc[i]["run"]}-{im_band.iloc[i]["ccd"]}.fits'

        if not os.path.exists(download_dir / ofile):
            unpack(str(download_dir / file))
        clean_image(str(download_dir / ofile))

        os.rename(download_dir / f"clean_{ofile}", raw_dir / f"clean_{ofile}")

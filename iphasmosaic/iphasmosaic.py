import asyncio
import os.path
import sys
import sqlite3
from pathlib import Path
from typing import Union, List, Optional
import logging

import aiofile
import aiohttp
from tqdm import tqdm

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.coordinates import SkyCoord
import astropy.units as u
from MontagePy.main import (
    mHdr,
    mImgtbl,
    mProjExec,
    mOverlaps,
    mDiffFitExec,
    mBgModel,
    mBgExec,
    mAdd,
)
from .utils import download_frames_by_5, orginize_images


FORMAT = "%(name)s - %(levelname)s:%(message)s"
LOGLEVEL = logging.DEBUG
logging.basicConfig(level=LOGLEVEL, format=FORMAT)
logger = logging.getLogger(__name__)


def _db_iphas_path() -> str:
    db_path = Path(
        os.path.join(os.path.dirname(sys._getframe(1).f_code.co_filename), "db/")
    )
    db_path = db_path / "iphas-images.sqlite"
    return str(db_path)


def _connect_iphas_db() -> sqlite3.Connection:
    """Connect to IPHAS database."""
    con = sqlite3.connect(_db_iphas_path())
    return con


def query_iphas_many(
    ra: float, dec: float, radius: float, band: str, qcgrade: str
) -> pd.DataFrame:
    """Querying the IPHAS database with a determined band and grade."""
    query = f"""
    SELECT ccd, run, band, ra, dec, ra_min, ra_max, dec_min, dec_max, seeing, qcgrade, utstart, depth
    FROM images
    WHERE band = "{band}" AND 
    """
    if isinstance(qcgrade, list):
        query += """( """
        qcgrade = [f'qcgrade="{x}"' for x in qcgrade]
        qcgrade = " OR ".join(qcgrade)
        query += qcgrade
        query += """ ) """
    else:
        query += '''qcgrade = "{qcgrade}"'''

    query_table = pd.read_sql(con=_connect_iphas_db(), sql=query)

    catalog = SkyCoord(query_table["ra"] * u.deg, query_table["dec"] * u.deg)
    c = SkyCoord(ra * u.deg, dec * u.deg)
    sep = c.separation(catalog)
    mask_sep = sep <= radius * u.arcmin

    query_table = query_table.loc[mask_sep].copy()

    return query_table


def _create_download_tasks(session, urls):
    tasks = []
    for url in urls:
        tasks.append(asyncio.create_task(session.get(url, ssl=False)))
    return tasks


def generate_iphas_url(run: str, ccd: str) -> str:
    """Download de IPHAS image using the run and the ccd numbers according to their webpage."""
    frame_url = f"http://www.iphas.org/data/images/r{run[:3]}/r{run}-{ccd}.fits.fz"

    return frame_url


def _remove_duplicates(input_df: pd.DataFrame, threshold: float = 10) -> pd.DataFrame:
    """Remove duplicated frames for an IPHAS query."""
    catalog = SkyCoord(input_df["ra"] * u.deg, input_df["dec"] * u.deg)
    input_df = input_df.assign(group=0)

    indexes = input_df.index.tolist()
    ind = 0
    current_group = 1
    while len(indexes) > ind:
        ra_current = input_df.loc[indexes[ind], "ra"] * u.deg
        dec_current = input_df.loc[indexes[ind], "dec"] * u.deg
        coord_current = SkyCoord(ra_current, dec_current)
        sep = coord_current.separation(catalog)
        mask_sep = sep <= threshold * u.arcsec
        if mask_sep.sum() > 1:
            input_df.loc[mask_sep, "group"] = current_group
            ind += mask_sep.sum()
        else:
            input_df.loc[indexes[ind], "group"] = current_group
            ind += 1
        current_group += 1

    input_df.drop_duplicates("group", keep="first", inplace=True)

    return input_df


class GetIPHASFrames:
    def __init__(self, ra: float, dec: float, radius: float) -> None:
        self.ra = ra * u.deg
        self.dec = dec * u.deg
        self.radius = radius * u.arcmin

    def get_all_frames(
        self,
        band: str,
        qcgrade: Union[List[str], str] = "A+",
        remove_duplicates: bool = True,
    ):
        table_query_frames = query_iphas_many(
            self.ra.value, self.dec.value, self.radius.value, band=band, qcgrade=qcgrade
        )

        if remove_duplicates:
            table_query_frames = _remove_duplicates(table_query_frames)

        return table_query_frames

    @staticmethod
    def download_frames(table: pd.DataFrame = None, download_dir: str = None) -> None:

        save_path = Path(download_dir)
        if download_dir is None:
            save_path = Path("./downloaded_frames")
        if not os.path.exists(str(save_path)):
            os.mkdir(str(save_path))

        urls = []
        for i in table.index.tolist():
            run = table.loc[i, "run"]
            ccd = table.loc[i, "ccd"]
            url = generate_iphas_url(str(run), str(ccd))

            if not Path.is_file(save_path / url.split("/")[-1]):
                urls.append(url)

        async def download():
            async with aiohttp.ClientSession() as session:
                tasks = _create_download_tasks(session, urls)
                responses = await asyncio.gather(*tasks)
                for response in tqdm(responses):
                    outfile = str(save_path / str(response.url).split("/")[-1])
                    assert response.status == 200
                    content = await response.read()
                    async with aiofile.async_open(outfile, "wb") as f:
                        await f.write(content)

        if len(urls) > 0:
            asyncio.run(download())


class IPHASMosaic(GetIPHASFrames):
    def __init__(
        self, name: str, band: str, ra: float, dec: float, radius: float
    ) -> None:
        super().__init__(ra, dec, radius)
        self.name = name
        self.band = band

    @property
    def cwd(self):
        return Path(os.getcwd()) / f"{self.name}_{self.band}"

    @property
    def path_tree(self) -> dict:
        wdir = self.cwd
        return {
            "wdir": wdir,
            "downloads": wdir / "downloads",
            "raw": wdir / "raw",
            "corrected": wdir / "corrected",
            "diffs": wdir / "diffs",
            "projected": wdir / "projected",
        }

    def create_tree(self) -> None:
        """Creates the directory tree."""
        if not os.path.exists(self.cwd):
            for dir_ in self.path_tree.values():
                os.makedirs(dir_)
        else:
            logger.warning(
                f"File {self.name} already exist. It will be set as a working directory."
            )
        logger.info(f"Current working directory: {self.cwd}")

    def download_and_prepare(
        self,
        qcgrade: Union[List[str], str] = "A++",
        remove_duplicates: bool = True,
        report_table: Optional[str] = None,
    ) -> None:
        logger.info("Preparing/reading images table.")
        if report_table is None:
            images_table = self.get_all_frames(self.band, qcgrade, remove_duplicates)
        else:
            images_table = pd.read_csv(report_table)
        images_table.to_csv(
            str(self.path_tree["wdir"] / f"report_{self.name}_{self.band}.csv"),
            mode="+w",
            index=False,
        )

        logger.info("Downloading images.")
        logger.info(f"Total images: {len(images_table)}")
        download_frames_by_5(
            images_table,
            GetIPHASFrames.download_frames,
            self.path_tree["downloads"],
        )

        logger.info(f"Organizing images in the working directory: {self.cwd}")
        orginize_images(
            images_table, self.band, self.path_tree["downloads"], self.path_tree["raw"]
        )

    def mosaic(self, fix_nan: bool = False):
        os.chdir(self.cwd)

        location = f"{self.ra.value} {self.dec.value} Equ J2000"
        size = round(self.radius.value / 60, 2)
        raw_folder = "raw"
        fitsname = f"mosaic_{self.name}_{self.band}.fits"

        rtn = mHdr(location, size, size, "region.hdr")
        rtn = mImgtbl(raw_folder, "rimages.tbl")
        logger.info("mImbtbl (raw):  " + str(rtn))

        rtn = mProjExec(
            raw_folder, "rimages.tbl", "region.hdr", projdir="projected", quickMode=True
        )
        logger.info("mProjExec:  " + str(rtn))

        rtn = mImgtbl("projected", "pimages.tbl")
        logger.info("mImgtbl (projected): " + str(rtn))

        rtn = mOverlaps("pimages.tbl", "diffs.tbl")
        logger.info("mOverlaps:    " + str(rtn))

        rtn = mDiffFitExec("projected", "diffs.tbl", "region.hdr", "diffs", "fits.tbl")
        logger.info("mDiffFitExec: " + str(rtn))

        rtn = mBgModel("pimages.tbl", "fits.tbl", "corrections.tbl")
        logger.info("mBgModel:     " + str(rtn))

        rtn = mBgExec("projected", "pimages.tbl", "corrections.tbl", "corrected")
        logger.info("mBgExec:             " + str(rtn))

        rtn = mImgtbl("corrected", "cimages.tbl")
        logger.info("mImgtbl (corrected): " + str(rtn))

        rtn = mAdd("corrected", "cimages.tbl", "region.hdr", fitsname)
        logger.info("mAdd:                " + str(rtn))

        if fix_nan:
            with fits.open(fitsname) as hdu:
                _, median, _ = sigma_clipped_stats(hdu[0].data, sigma=2.8)
                nan_mask = np.isnan(hdu[0].data)
                hdu[0].data[nan_mask] = median

                hdu.writeto(fitsname, overwrite=True)


if __name__ == "__main__":
    c = SkyCoord("00:25:52.71", "+59:57:34.0", unit=("hourangle", "deg"))
    # c = SkyCoord.from_name("PN G132.8+02.0")

    ipm = IPHASMosaic("IP002552", "halpha", c.ra.value, c.dec.value, 30)
    ipm.create_tree()
    ipm.download_and_prepare(["A++", "A+"], True, None)
    ipm.mosaic(fix_nan=True)

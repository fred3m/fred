import numpy as np
import ipywidgets as widgets
from IPython.display import display

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

import lsst.afw.table
from lsst.afw.image import MultibandExposure
from scarlet.display import AsinhMapping, img_to_rgb
import lsst.meas.extensions.scarlet as mes

class InteractivePlot:
    """A matplotlib plot will drilldown capabilities

    In order to follow-up with outliers this class is intended to
    be inherited by different types of plots. When a data point in
    the plot is clicked, the source corresponding to the selected
    data point has its image and data displayed and stored within
    this class.
    
    This class requires ineritants to define the following properties:
    
    Attributes
    ----------
    df: `pandas.DataFrame`
        The `DataFrame` that contains the entire pipe_analysis catalog
    tract: int
        The tract that contains the source
    selected: array of int
        Array with the indices from `df` for all selected sources
        (if the user clicked on a point in a histogram there can
        be multiple selected sources, otherwise a single source
        is selected).
    selectIndex: int
        The index of the currently selected source in `selected`.
    butler: `daf.persistence.Butler`
        The Butler that points the the location of the image and
        catalog data.
    lblStatus: `ipywidgets.Label`
        The current status. This updates the user as different
        data products are loaded.
    filters: string
        Names of the filters to load (typically "grizy").
        The case does not matter, as the filters are converted into
        ``"HSC-{}".format(f)``, where ``f`` is the name of the filter.
    stretch: double
        The ``stretch`` parameter for the asinh mapping
    Q: double
        The ``Q`` parameter for the asinh mapping
    fig: `matplotlib.Figure`
        The figure that displays both the plot and the image of the blend.
    ax: list of matplotlib axes
        `ax[0]` is expected to be the interactive plot while
        `ax[1]` is the source image.
    coadds: `lsst.afw.MultibandExposure`
        The multiband coadd for the selected source
    footprint: `lsst.detection.Footprint`
        The footprint of the selected sources' parent
    band: str
        The name of the band for the ``meas`` catalog for the
        selected source. This should be the full filter name,
        for example ``HSC-R``.
    peak: `PeakCatalog`
        The peak catalog row for the selected source
    """
    def loadImage(self):
        """Load and display the parent blend of the selected source.
        """
        self.lblStatus.value = "Loading parent catalog"
        # Load the parent blend of the currently selected source
        row = self.df.iloc[self.selected].iloc[self.selectIndex]
        patch = row["patchId"]
        dataId = {
            "tract": self.tract,
            "patch": patch,
        }
        mergeDet = self.butler.get("deepCoadd_mergeDet", dataId)
        pid = row["parent"]
        if pid == 0:
            pid = row["id"]
        src = mergeDet[mergeDet["id"]==pid][0]

        self.lblStatus.value = "Loading parent MultibandExposure"
        # Extract the bounding box off the parent and extract the MultibandExposure
        fp = src.getFootprint()
        self.footprint = fp
        bbox = fp.getBBox()
        coadds = [
            self.butler.get(
                "deepCoadd_calexp_sub",
                dataId,
                bbox=bbox,
                filter="HSC-{}".format(f.upper()))
            for f in self.filters
        ]
        coadds = MultibandExposure.fromExposures(self.filters, coadds)
        self.coadds = coadds
        
        # Set the visualization parameters
        self.vmin = 0
        self.vmax = np.max(coadds.image.array)
        
        meas = self.butler.get(
            "deepCoadd_meas",
            dataId,
            flags=lsst.afw.table.SOURCE_IO_NO_FOOTPRINTS,
            filter=self.band
        )
        self.peak = meas[meas["id"]==row["id"]][0]
        
        self.displayImage()
    
    def coaddToRGB(self):
        """Convert a coadd image to an RGB image
        """
        images = self.coadds.image.array
        norm = AsinhMapping(minimum=self.vmin, stretch=self.vmax*self.stretch, Q=self.Q)
        rgb = img_to_rgb(images, norm=norm)
        # Apply a mask to only display the pixels in the footprint
        mask = mes.scarletDeblendTask.getFootprintMask(self.footprint, self.coadds)
        rgb = np.dstack([rgb, ~mask*255])
        return rgb
    
    def displayImage(self):
        """Display the image of the blend with the sources marked
        """
        self.lblStatus.value = "Displaying image"
        # Display the image
        rgb = self.coaddToRGB()
        self.ax[1].clear()
        self.image = self.ax[1].imshow(rgb, origin="lower")
        self.ax[1].set_title("source {} of {}".format(self.selectIndex+1, len(self.selected)))

        # Plot all of the sources in the blend
        self.lblStatus.value = "Plotting sources"
        bbox = self.footprint.getBBox()
        xmin = bbox.getMinX()
        ymin = bbox.getMinY()
        for pk in self.footprint.peaks:
            self.ax[1].plot(pk["i_x"]-xmin, pk["i_y"]-ymin, "wx", mew=2)

        # Plot the currently selected source
        self.ax[1].plot(
            self.peak["deblend_peak_center_x"]-xmin,
            self.peak["deblend_peak_center_y"]-ymin,
            "cx",
            mew=2
        )
        plt.tight_layout()
        self.fig.canvas.draw()
        
        self.lblStatus.value = "selected source at {}".format(
            (self.peak["deblend_peak_center_x"]-xmin,
             self.peak["deblend_peak_center_y"]-ymin)
        )
        lbl = "Tract {}, patch {}, parent {}"
        self.lblSelect.value = lbl.format(self.tract, patch, pid)
    
    def updateImage(self):
        """Quickly update the image without redrawing the axis
        
        This is typically done when updating vmin, vmax, Q, or stretch
        """
        rgb = self.coaddToRGB()
        self.image.set_data(rgb)
        self.fig.canvas.draw()
        
class InteractiveHist(InteractivePlot):
    """An interactive histogram
    
    Histogram version of a pipe_analysis plot, which
    allows the user to slect a data point and select all
    of the sources contained in that data point.
    
    In addition to the attributes listed below, it also
    inherits the attributes from `InteractivePlot`.
    
    Attributes
    ----------
    hist: array
        The histogram that is plotted
    xedges: array
        The values of the x column for the edges of each bin
    yedges: array
        The values of the y column for the edges of each bin
    rect: `matplotlib.patches.Rectangle`
        The rectangle that shows the selected bin
    colorbar: `matplotlib.Colorbar`
        The colorbar for the histogram
    lblCursor: `ipywidgets.Label`
        The label that shows the number of sources in the
        bin that is underneath the cursor.
    lblSelect: `ipywidgets.Label`
        The label that shows the number of sources in the
        bin that has been selected.
    
    """
    def __init__(self, butler, df=None, tract=9813, filters=None, band=None, cursorColor=None):
        if filters is None:
            filters = "grizy"
        if band is None:
            band = "HSC-R"
        if cursorColor is None:
            cursorColor = "#c9392e"

        self.butler = butler
        self.tract=tract        
        self.filters = filters
        self.band = band
        self.cursorColor = cursorColor
        
        if df is None:
            cat = butler.get(
                "analysisCoaddTable_forced",
                tract=tract,
                filter=band,
                subdir=""
            )

            df = cat.toDataFrame()
            sources = (
                df["parent"] != 0
                & df["detect_isPatchInner"]
                & ~df["merge_peak_sky"]
                & np.isfinite(df["base_PsfFlux_instFlux"])
                & (df["base_PsfFlux_instFlux"] > 0)
                & np.isfinite(df["modelfit_CModel_instFlux"])
                & (df["modelfit_CModel_instFlux"] > 0)
            )
            df = df[sources]
        self.df = df
        self.selectIndex = 0
        self.Q = 10
        self.stretch = 0.005
    
    def previousSource(self, event):
        """Load the parent blend for the previous source
        """
        if len(self.selected) <= 1:
            # No need to do anything if there is only one selected source
            return
        
        self.selectIndex -= 1
        if self.selectIndex < 0:
            self.selectIndex = len(self.selected) - 1
        self.loadImage()
        
    def nextSource(self, event):
        """Load the next parent blend for the previous source
        """
        self.lblStatus.value = "NEXT"
        if len(self.selected) <= 1:
            # No need to do anything if there is only one selected source
            return
        
        self.selectIndex += 1
        if self.selectIndex > len(self.selected)-1:
            self.selectIndex = 0
        self.loadImage()
    
    def updateQ(self, change):
        """Update the 'Q' parameter in the asinh algorithm
        """
        self.Q = change["new"]
        self.updateImage()
    
    def updateStretch(self, change):
        """Update the 'stretch' parameter in the asinh algorithm
        """
        self.stretch = change["new"]
        self.updateImage()
    
    def initControls(self):
        """Initialize the navigation controls and sliders
        """
        # Initialize and display the navigation buttons
        previousButton = widgets.Button(description="<")
        previousButton.on_click(self.previousSource)
        nextButton = widgets.Button(description=">")
        nextButton.on_click(self.nextSource)
        display(widgets.HBox([previousButton, nextButton]))
        
        # Initialize and display the parameters for the asinh mapping
        sliderQ = widgets.FloatSlider(value=10, min=0, max=100, step=.5, readout_format='.1f')
        sliderQ.observe(self.updateQ, names="value")
        display(widgets.HBox([widgets.Label("Q"), sliderQ]))
        sliderStretch = widgets.FloatSlider(value=0.005, min=0.00001, max=0.1, step=0.0001, readout_format='.4f')
        sliderStretch.observe(self.updateStretch, names="value")
        display(widgets.HBox([widgets.Label("stretch"), sliderStretch]))
    
    def setColumnX(self, column, toMag=False):
        """Based on the column name set the x column
        
        If ``toMag`` is ``True`` then convert the (flux) values
        in the column into magnitudes
        """
        assert column is not None
        self.x = self.df[column]
        self.xColumn = column
        if toMag:
            self.x = -2.5*np.log10(self.x)
        self.xColumn = column
        return self.x
    
    def setColumnY(self, column, toMag=False):
        """Based on the column name set the y column
        
        If ``toMag`` is ``True`` then convert the (flux) values
        in the column into magnitudes
        """
        assert column is not None
        self.y = self.df[column]
        self.yColumn = column
        if toMag:
            self.y = -2.5*np.log10(self.y)
        self.yColumn = column
        return self.x
    
    def initHist(self, cuts=None, xColumn=None, yColumn=None, width=100, ratio=0.5,
                 close=True, xMag=False, yMag=False, norm=None, figsize=None):
        """Initialize the histogram plot
        
        Parameters
        ----------
        cuts: array
            The cuts (rows to select) in the x and y columns
        xColumn: str
            The name of the x column to plot
        yColumn: str
            The name of the y column to plot
        width: int
            The number of column in the histogram
        ratio: int
            The ratio of the number of rows/number of columns
            (eg. y/x)
        close: bool
            Whether or not to close all open plots before creating the figure.
            This is recommended, since matplotlib will keep all of the plots open
            by default, even if the same cell is rerun, which can result in
            significant memory consumption.
        xMag: bool
            Whether or not the x column is a flux value
            that must be converted to magnitudes.
        yMag: bool
            Whether or not the y column is a flux value
            that must be converted to magnitudes.
        norm: `matplotlib.colors.Normalize`
            The type of normalization to use for the histogram colors.
            By default this is a log norm, since we expect the bulk of the
            data points to lie near the median, making it easier to identify
            the outliers in any given bin.
        figsize: tuple
            The size of the figure that contains both the histogram and blend image
        """
        if figsize is None:
            figsize = (10, 15)
        if xColumn is not None:
            self.setColumnX(xColumn, xMag)
        if yColumn is not None:
            self.setColumnY(yColumn, yMag)
        if norm is None:
            norm = matplotlib.colors.LogNorm()

        if close:
            # Close all active plots (to prevent wasting memory)
            plt.close('all')
        
        if cuts is not None:
            x = self.x[cuts]
            y = self.y[cuts]
        else:
            x = self.x
            y = self.y

        # Create the figure
        self.fig = plt.figure(figsize=figsize)
        # The first axis is the histogram,
        # the second is the cutout image
        self.ax = [self.fig.add_subplot(2, 1, n+1) for n in range(2)]
        
        self.hist, self.xedges, self.yedges, im = self.ax[0].hist2d(
            x,
            y,
            bins=[width, int(width*ratio)],
            cmin=1,
            norm=norm,
        )
        # Create the cursor (with zero size, so it is not drawn)
        self.rect = patches.Rectangle((np.median(x), np.median(y)), 0, 0, color=self.cursorColor)
        self.ax[0].add_patch(self.rect)
        self.ax[0].set_xlabel(self.xColumn)
        self.ax[0].set_ylabel(self.yColumn)
        # Create a placeholder for the blend image
        self.ax[1].imshow(np.zeros((10,10)))
        
        # Create the colorbar
        divider = make_axes_locatable(self.ax[0])
        cax = divider.append_axes('top', size='5%', pad="10%")
        self.fig.colorbar(im, cax=cax, orientation='horizontal')
        
        # Create the label for the cursor value
        self.lblCursor = widgets.Label("cursor:")
        # Create the label for the selected value
        self.lblSelect = widgets.Label("selected:")
        # Create the label for the status indicator
        self.lblStatus = widgets.Label("")
        
        # The currently selected indices
        self.selected = np.array([])

        self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.fig.canvas.mpl_connect('motion_notify_event', self.onmove)

        plt.tight_layout()
        plt.show()
        
        # Create the buttons to navigate between sources
        self.initControls()
        
        display(self.lblCursor)
        display(self.lblSelect)
        display(self.lblStatus)

    def onmove(self, event):
        """Actions to be performed when the cursor is moved over the plot
        
        In this case it updates the value of the `lblCursor`, which
        shows the number of sources in the bin that is underneath the cursor.
        """
        # Make sure that the cursor is in the data region
        if event.xdata is None or event.ydata is None:
            return
        # Find the datapoints within the current region
        ix = np.where((event.xdata > self.xedges[:-1]) & (event.xdata < self.xedges[1:]))[0]
        iy = np.where((event.ydata > self.yedges[:-1]) & (event.ydata < self.yedges[1:]))[0]
        if len(ix) ==0 or len(iy) == 0:
            # The cursor is on a pixel that has no sources
            self.lblCursor.value = "cursor: 0"
            return

        # Set the value of the current location
        value = self.hist[ix[0], iy[0]]
        if not np.isfinite(value):
            value = 0
        value = int(value)
        
        self.lblCursor.value = "cursor: {}".format(value)

    def onclick(self, event):
        """Actions to be performed when the plot is clicked
        
        This updates the `lblSelect` label and loads the image
        of the parent blend.
        """
        self.lblStatus.value = "clicked"
        self.selectIndex
        # Find the datapoints within the region
        ix = np.where((event.xdata > self.xedges[:-1]) & (event.xdata < self.xedges[1:]))[0]
        iy = np.where((event.ydata > self.yedges[:-1]) & (event.ydata < self.yedges[1:]))[0]
        if len(ix) ==0 or len(iy) == 0:
            # The cursor is on a pixel that has no sources
            self.lblSelect.value = "cursor: 0"
            return

        width = np.abs(self.xedges[ix][0] - self.xedges[ix+1][0])
        height = np.abs(self.yedges[iy][0] - self.yedges[iy+1][0])

        value = self.hist[ix[0], iy[0]]
        if not np.isfinite(value):
            value = 0
        value = int(value)

        if value == 0:
            self.lblStatus.value = "No sources in the selected region"
            self.clearCursor(self.rect)
            return

        self.rect.set_xy((self.xedges[ix][0], self.yedges[iy][0]))
        self.rect.set_width(width)
        self.rect.set_height(height)

        left = self.xedges[ix][0]
        right = self.xedges[ix+1][0]
        bottom = self.yedges[iy][0]
        top = self.yedges[iy+1][0]
        
        self.extent = (left, right, bottom, top)

        _x = np.array(self.x)
        _y = np.array(self.y)

        c = (
            (_x > left) & 
            (_x < right) &
            (_y > bottom) &
            (_y < top)
        )

        self.selected = np.where(c)[0]
        self.lblSelect.value = "selected: {}".format(len(self.selected))
        
        plt.tight_layout()
        self.fig.canvas.draw()
        
        self.lblStatus.value = "loading image"

        self.loadImage()
    
    def clearCursor(rect):
        """Clear the cursor marker if there is no selected bin
        """
        self.rect.set_width(0)
        self.rect.set_height(0)

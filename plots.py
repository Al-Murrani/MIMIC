import plotly.express as px


class PlotlyPlots:
    def __init__(self, data_frame):
        self.data = data_frame
        """
        Initialize with a pandas data frame.

        data_frame parameter is the data used for plotting.
        """

    def plot(self, plot_type, **kwargs):
        """
        Create a plot based on the specified plot.
        plot_type (str): the type of the plot to create.
        **kwargs:additional argument to customize the plot.  These should match the one excepted by the plot.
        """
        plot_function = {
            'bar': px.bar,
            'scatter': px.scatter
        }.get(plot_type)

        if plot_function is None:
            raise ValueError(f"Unsupported plot type: {plot_type}")

        fig = plot_function(self.data, **kwargs)

        fig.update_layout(title_x=0.5)
        fig.show()

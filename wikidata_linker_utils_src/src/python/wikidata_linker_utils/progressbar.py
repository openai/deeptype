import progressbar

percentage = progressbar.Percentage()
counter = progressbar.Counter()
bar = progressbar.Bar()
adaptive_eta = progressbar.AdaptiveETA()


class MessageProgressbar(progressbar.ProgressBar):
    def set_message(self, message):
        self.widgets[0] = message + " "

    def set_item(self, item):
        self.widgets[4] = " %s) " % (item,)


def get_progress_bar(message, max_value=None, item="lines"):
    """
    Construct a progressbar iterator wrapper
    with an ETA and percentage information.

    Arguments:
    ----------
        message : str, title for the progress bar.
        max_value : None or int

    Returns:
    --------
        ProgressBar : object that can wrap an iterator
            and print out duration estimates and
            iteration stats.
    """
    widgets = [
        message + " ",
        percentage,
        " (",
        counter,
        " %s) " % (item,),
        bar,
        adaptive_eta
    ]
    return MessageProgressbar(widgets=widgets, maxval=max_value)

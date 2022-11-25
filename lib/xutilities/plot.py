def autolabel(rects, ax, text=None):
    # Get y-axis height to calculate label position from.
    (y_bottom, y_top) = ax.get_ylim()
    y_height = y_top - y_bottom

    for i, rect in enumerate(rects):
        height = rect.get_height()
        if text is not None:
            this_text = text if type(text) is str else text[i]
        else:
            this_text = '%d' % int(height)

        # Fraction of axis height taken up by this rectangle
        p_height = (height / y_height)

        # If we can fit the label above the column, do that;
        # otherwise, put it inside the column.
        if p_height > 0.95: # arbitrary; 95% looked good to me.
            label_position = height - (y_height * 0.05)
        else:
            label_position = height + (y_height * 0.01)

        ax.text(rect.get_x() + rect.get_width()/2., label_position,
                this_text,
                ha='center', va='bottom')
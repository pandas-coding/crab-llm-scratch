use std::path::Path;
use plotly::layout::Axis;
use plotly::Plot;

/// Plotting the classification loss
pub fn plot_values<P: AsRef<Path>>(
    epoches_seen: Vec<f32>,
    examples_seen: Vec<f32>,
    train_values: Vec<f32>,
    val_values: Vec<f32>,
    label: &str,
    save_path: P,
) -> candle_core::Result<()> {
    let trace1 = plotly::Scatter::new(epoches_seen.clone(), train_values.clone())
        .show_legend(false)
        .opacity(0f64)
        .mode(plotly::common::Mode::LinesMarkers);
    let trace2 = plotly::Scatter::new(epoches_seen.clone(), val_values.clone())
        .show_legend(false)
        .opacity(0f64)
        .mode(plotly::common::Mode::LinesMarkers);
    let trace3 = plotly::Scatter::new(examples_seen.clone(), train_values)
        .name(format!("Training {label}").as_str())
        .x_axis("x2")
        .mode(plotly::common::Mode::LinesMarkers);
    let trace4 = plotly::Scatter::new(examples_seen, val_values)
        .name(format!("Validation {label}"))
        .x_axis("x2")
        .mode(plotly::common::Mode::LinesMarkers);

    let layout = plotly::Layout::new()
        .x_axis(Axis::new().title("Epochs"))
        .x_axis2(Axis::new().title("Examples Seen").side(plotly::common::AxisSide::Top));

    let mut plot = Plot::new();
    plot.add_trace(trace1);
    plot.add_trace(trace2);
    plot.add_trace(trace3);
    plot.add_trace(trace4);
    plot.set_layout(layout);
    plot.write_html(save_path);
    Ok(())
}

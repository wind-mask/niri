use client::ClientId;
use insta::assert_snapshot;
use smithay::utils::Point;
use wayland_client::protocol::wl_surface::WlSurface;

use super::*;
use crate::layout::LayoutElement as _;

// Sets up a fixture with two outputs and 100×100 window.
fn set_up() -> (Fixture, ClientId, WlSurface) {
    let mut f = Fixture::new();
    f.add_output(1, (1920, 1080));
    f.add_output(2, (1280, 720));

    let id = f.add_client();
    let window = f.client(id).create_window();
    let surface = window.surface.clone();
    window.commit();
    f.roundtrip(id);

    let window = f.client(id).window(&surface);
    window.attach_new_buffer();
    window.set_size(100, 100);
    window.ack_last_and_commit();
    f.double_roundtrip(id);

    (f, id, surface)
}

#[test]
fn windowed_fullscreen() {
    let (mut f, id, surface) = set_up();

    let _ = f.client(id).window(&surface).recent_configures();

    let niri = f.niri();
    let mapped = niri.layout.windows().next().unwrap().1;
    let window_id = mapped.window.clone();

    // Enable windowed fullscreen.
    niri.layout.toggle_windowed_fullscreen(&window_id);
    f.double_roundtrip(id);

    // Should request fullscreen state with the tiled size.
    let window = f.client(id).window(&surface);
    assert_snapshot!(
        window.format_recent_configures(),
        @"size: 936 × 1048, bounds: 1888 × 1048, states: [Activated, Fullscreen]"
    );

    let mapped = f.niri().layout.windows().next().unwrap().1;
    // Not committed yet.
    assert!(!mapped.is_windowed_fullscreen());

    // Commit in response.
    let window = f.client(id).window(&surface);
    window.ack_last_and_commit();
    f.roundtrip(id);

    let mapped = f.niri().layout.windows().next().unwrap().1;
    // Now it is committed.
    assert!(mapped.is_windowed_fullscreen());

    // Disable windowed fullscreen.
    f.niri().layout.toggle_windowed_fullscreen(&window_id);
    f.double_roundtrip(id);

    // Should request without fullscreen state with the tiled size.
    let window = f.client(id).window(&surface);
    assert_snapshot!(
        window.format_recent_configures(),
        @"size: 936 × 1048, bounds: 1888 × 1048, states: [Activated]"
    );

    let mapped = f.niri().layout.windows().next().unwrap().1;
    // Not committed yet.
    assert!(mapped.is_windowed_fullscreen());

    // Commit in response.
    let window = f.client(id).window(&surface);
    window.ack_last_and_commit();
    f.roundtrip(id);

    let mapped = f.niri().layout.windows().next().unwrap().1;
    // Now it is committed.
    assert!(!mapped.is_windowed_fullscreen());
}

#[test]
fn windowed_fullscreen_chain() {
    let (mut f, id, surface) = set_up();

    let _ = f.client(id).window(&surface).recent_configures();

    let mapped = f.niri().layout.windows().next().unwrap().1;
    let window_id = mapped.window.clone();

    f.niri().layout.toggle_windowed_fullscreen(&window_id);
    f.roundtrip(id);
    f.niri().layout.toggle_windowed_fullscreen(&window_id);
    f.roundtrip(id);
    f.niri().layout.toggle_windowed_fullscreen(&window_id);
    f.roundtrip(id);
    f.niri().layout.toggle_windowed_fullscreen(&window_id);
    f.double_roundtrip(id);

    // Should be four configures matching the four requests.
    let window = f.client(id).window(&surface);
    assert_snapshot!(
        window.format_recent_configures(),
        @r"
    size: 936 × 1048, bounds: 1888 × 1048, states: [Activated, Fullscreen]
    size: 936 × 1048, bounds: 1888 × 1048, states: [Activated]
    size: 936 × 1048, bounds: 1888 × 1048, states: [Activated, Fullscreen]
    size: 936 × 1048, bounds: 1888 × 1048, states: [Activated]
    "
    );

    let window = f.client(id).window(&surface);
    let serials = Vec::from_iter(
        window.configures_received[window.configures_received.len() - 4..]
            .iter()
            .map(|(s, _c)| *s),
    );

    let get_state = |f: &mut Fixture| {
        let mapped = f.niri().layout.windows().next().unwrap().1;
        format!(
            "fs {}, wfs {}",
            mapped.sizing_mode().is_fullscreen(),
            mapped.is_windowed_fullscreen()
        )
    };

    let mut states = vec![get_state(&mut f)];
    for serial in serials {
        let window = f.client(id).window(&surface);
        window.xdg_surface.ack_configure(serial);
        window.commit();
        f.roundtrip(id);
        states.push(get_state(&mut f));
    }

    // We expect fs to always be false (because each Fullscreen state request corresponded to a
    // windowed fullscreen), and wfs to toggle on and off.
    assert_snapshot!(
        states.join("\n"),
        @r"
    fs false, wfs false
    fs false, wfs true
    fs false, wfs false
    fs false, wfs true
    fs false, wfs false
    "
    );
}

#[test]
fn unfullscreen_before_fullscreen_ack_doesnt_prevent_view_offset_save_restore() {
    let (mut f, id, _surface) = set_up();

    let window2 = f.client(id).create_window();
    let surface2 = window2.surface.clone();
    window2.commit();
    f.roundtrip(id);

    let window2 = f.client(id).window(&surface2);
    window2.attach_new_buffer();
    window2.set_size(200, 200);
    window2.ack_last_and_commit();
    f.double_roundtrip(id);

    let _ = f.client(id).window(&surface2).recent_configures();

    let niri = f.niri();
    let mapped2 = niri.layout.windows().last().unwrap().1;
    let window2_id = mapped2.window.clone();

    // The view position is at the first window.
    assert_snapshot!(niri.layout.active_workspace().unwrap().scrolling().view_pos(), @"-16");

    // Fullscreen window2 and send the configure so we can clear pending.
    niri.layout.set_fullscreen(&window2_id, true);
    f.double_roundtrip(id);

    // Before acking, unfullscreen the column, clearing the pending fullscreen flag.
    f.niri().layout.set_fullscreen(&window2_id, false);

    // Now, window2 receives the fullscreen configure and resizes in response.
    let window2 = f.client(id).window(&surface2);
    assert_snapshot!(
        window2.format_recent_configures(),
        @"size: 1920 × 1080, bounds: 1888 × 1048, states: [Activated, Fullscreen]"
    );
    let (_, configure) = window2.configures_received.last().unwrap();
    window2.set_size(configure.size.0 as u16, configure.size.1 as u16);
    window2.ack_last_and_commit();
    f.double_roundtrip(id);
    f.niri_complete_animations();

    // The view position is now at the fullscreen-sized window2.
    assert_snapshot!(f.niri().layout.active_workspace().unwrap().scrolling().view_pos(), @"116");

    // Now, window2 receives the unfullscreen configure and resizes in response.
    let window2 = f.client(id).window(&surface2);
    assert_snapshot!(
        window2.format_recent_configures(),
        @"size: 936 × 1048, bounds: 1888 × 1048, states: [Activated]"
    );
    window2.set_size(200, 200);
    window2.ack_last_and_commit();
    f.roundtrip(id);
    f.niri_complete_animations();

    // The view position should restore to the first window.
    assert_snapshot!(f.niri().layout.active_workspace().unwrap().scrolling().view_pos(), @"-16");
}

#[test]
fn interactive_move_unfullscreen_to_scrolling_restores_size() {
    let (mut f, id, surface) = set_up();

    let _ = f.client(id).window(&surface).recent_configures();

    let niri = f.niri();
    let mapped = niri.layout.windows().next().unwrap().1;
    let window = mapped.window.clone();
    niri.layout.set_fullscreen(&window, true);
    f.double_roundtrip(id);

    // This should request a fullscreen size.
    assert_snapshot!(
        f.client(id).window(&surface).format_recent_configures(),
        @"size: 1920 × 1080, bounds: 1888 × 1048, states: [Activated, Fullscreen]"
    );

    // Start an interactive move which causes an unfullscreen.
    let output = f.niri_output(1);
    let niri = f.niri();
    let mapped = niri.layout.windows().next().unwrap().1;
    let window = mapped.window.clone();
    niri.layout
        .interactive_move_begin(window.clone(), &output, Point::default());
    niri.layout.interactive_move_update(
        &window,
        Point::from((1000., 0.)),
        output,
        Point::default(),
    );
    f.double_roundtrip(id);

    // This should request the tiled size.
    assert_snapshot!(
        f.client(id).window(&surface).format_recent_configures(),
        @"size: 936 × 1048, bounds: 1920 × 1080, states: [Activated]"
    );
}

#[test]
fn interactive_move_unmaximize_to_scrolling_restores_size() {
    let (mut f, id, surface) = set_up();

    let _ = f.client(id).window(&surface).recent_configures();

    let niri = f.niri();
    let mapped = niri.layout.windows().next().unwrap().1;
    let window = mapped.window.clone();
    niri.layout.set_maximized(&window, true);
    f.double_roundtrip(id);

    // This should request a maximized size.
    assert_snapshot!(
        f.client(id).window(&surface).format_recent_configures(),
        @"size: 1920 × 1080, bounds: 1888 × 1048, states: [Activated, Maximized]"
    );

    // Start an interactive move which causes an unmaximize.
    let output = f.niri_output(1);
    let niri = f.niri();
    let mapped = niri.layout.windows().next().unwrap().1;
    let window = mapped.window.clone();
    niri.layout
        .interactive_move_begin(window.clone(), &output, Point::default());
    niri.layout.interactive_move_update(
        &window,
        Point::from((1000., 0.)),
        output,
        Point::default(),
    );
    f.double_roundtrip(id);

    // This should request the tiled size.
    assert_snapshot!(
        f.client(id).window(&surface).format_recent_configures(),
        @"size: 936 × 1048, bounds: 1920 × 1080, states: [Activated]"
    );
}

/// With the magnify upscaling mode, going fullscreen must not resize the client: niri draws its
/// content scaled up into the output's area instead.
#[test]
fn magnify_fullscreen_keeps_client_size() {
    let mut config = niri_config::Config::default();
    config.window_rules.push(niri_config::WindowRule {
        upscaling: niri_config::UpscalingRule {
            mode: Some(niri_config::UpscalingMode::Magnify),
            ..Default::default()
        },
        ..Default::default()
    });

    let mut f = Fixture::with_config(config);
    f.add_output(1, (1920, 1080));

    let id = f.add_client();
    let window = f.client(id).create_window();
    let surface = window.surface.clone();
    window.commit();
    f.roundtrip(id);

    let window = f.client(id).window(&surface);
    window.attach_new_buffer();
    window.set_size(100, 100);
    window.ack_last_and_commit();
    f.double_roundtrip(id);

    let _ = f.client(id).window(&surface).recent_configures();

    let window_id = f.niri().layout.windows().next().unwrap().1.window.clone();

    f.niri().layout.toggle_fullscreen(&window_id);
    f.double_roundtrip(id);

    // The client is told that it is fullscreen, but keeps its own size.
    let window = f.client(id).window(&surface);
    assert_snapshot!(
        window.format_recent_configures(),
        @"size: 100 × 100, bounds: 1888 × 1048, states: [Activated, Fullscreen]"
    );

    window.ack_last_and_commit();
    f.double_roundtrip(id);

    // The layout still gives the window the whole output, and the content is drawn magnified.
    let mapped = f.niri().layout.windows().next().unwrap().1;
    let visual = mapped.visual_size();
    assert_ne!(
        visual,
        mapped.size(),
        "expected the window to occupy more than its own size"
    );

    let magnify = mapped
        .magnify_geometry()
        .expect("expected magnify geometry");
    assert_eq!(magnify.source, smithay::utils::Size::from((100., 100.)));
    assert_eq!(magnify.target, visual.to_f64());
    // A 100×100 window in a wide area is limited by height, so it gets bars on the sides.
    assert!(magnify.scale > 1.);
    assert!(magnify.content.loc.x > 0.);

    // Leaving fullscreen goes back to drawing the window at its own size, and the client was never
    // resized.
    f.niri().layout.toggle_fullscreen(&window_id);
    f.double_roundtrip(id);

    let window = f.client(id).window(&surface);
    window.ack_last_and_commit();
    f.double_roundtrip(id);

    let mapped = f.niri().layout.windows().next().unwrap().1;
    assert_eq!(mapped.magnify_geometry(), None);
    assert_eq!(mapped.visual_size(), mapped.size());
    assert_eq!(mapped.size(), smithay::utils::Size::from((100, 100)));
}

/// The pointer is sent to a magnified window in the client's own coordinates, even though its
/// content is drawn scaled.
///
/// The client is sent the pointer position minus the position of the surface, and the magnified
/// content is not a fixed offset from the pointer: the position that goes with the surface has to
/// be derived from where the pointer is.
#[test]
fn magnify_maps_pointer_into_the_client() {
    let mut config = niri_config::Config::default();
    config.window_rules.push(niri_config::WindowRule {
        upscaling: niri_config::UpscalingRule {
            mode: Some(niri_config::UpscalingMode::Magnify),
            ..Default::default()
        },
        ..Default::default()
    });

    let mut f = Fixture::with_config(config);
    f.add_output(1, (1920, 1080));

    let id = f.add_client();
    let window = f.client(id).create_window();
    let surface = window.surface.clone();
    window.commit();
    f.roundtrip(id);

    let window = f.client(id).window(&surface);
    window.attach_new_buffer();
    window.set_size(100, 100);
    window.ack_last_and_commit();
    f.double_roundtrip(id);
    let _ = f.client(id).window(&surface).recent_configures();

    let window_id = f.niri().layout.windows().next().unwrap().1.window.clone();
    f.niri().layout.toggle_fullscreen(&window_id);
    f.double_roundtrip(id);
    f.client(id).window(&surface).ack_last_and_commit();
    f.double_roundtrip(id);

    // The drawn geometry and the area the layout gave the window have to agree for this to measure
    // the content, and they do once the fullscreen animation has run its course.
    f.niri_complete_animations();

    // The letterbox bars do not take input, so the window is under the pointer exactly within the
    // drawn content. Scanning for that finds where the content ends up.
    let over_window = |f: &mut Fixture, x: f64, y: f64| {
        f.niri()
            .contents_under(Point::from((x, y)))
            .surface
            .is_some()
    };

    let (mut left, mut right) = (None, None);
    for x in 0..1920 {
        let x = f64::from(x) + 0.5;
        if over_window(&mut f, x, 540.) {
            left.get_or_insert(x);
            right = Some(x);
        }
    }
    let (left, right) = (left.expect("no window found"), right.unwrap());
    let width = right - left;

    let center_x = left + width / 2.;
    let (mut top, mut bottom) = (None, None);
    for y in 0..1080 {
        let y = f64::from(y) + 0.5;
        if over_window(&mut f, center_x, y) {
            top.get_or_insert(y);
            bottom = Some(y);
        }
    }
    let (top, bottom) = (top.expect("no window found"), bottom.unwrap());
    let height = bottom - top;

    // The window is square and the area is wide, so the content is limited by height.
    assert!(
        (width - height).abs() <= 2.,
        "expected square magnified content, got {width}×{height}"
    );

    let half = height / 2.;
    let quarter = left + width / 4.;

    // A quarter of the way into the content is a quarter of the way into the client's buffer, and
    // the vertical middle is its middle, no matter how much the content is magnified.
    let point = Point::from((quarter, top + half));
    let (_, surface_pos) = f.niri().contents_under(point).surface.unwrap();
    let client_point = point - surface_pos;

    assert!(
        (client_point.x - 25.).abs() <= 1.5,
        "expected x = 25, got {client_point:?} for {point:?} in content {left}..{right}"
    );
    assert!(
        (client_point.y - 50.).abs() <= 1.5,
        "expected y = 50, got {client_point:?} for {point:?} in content {top}..{bottom}"
    );
}

/// The magnify mode is meant for floating windows: a small floating window that is put in
/// fullscreen fills the output with magnified content, while the client keeps its size.
#[test]
fn magnify_floating_fullscreen_keeps_client_size() {
    let mut config = niri_config::Config::default();
    config.window_rules.push(niri_config::WindowRule {
        upscaling: niri_config::UpscalingRule {
            mode: Some(niri_config::UpscalingMode::Magnify),
            ..Default::default()
        },
        ..Default::default()
    });

    let mut f = Fixture::with_config(config);
    f.add_output(1, (1920, 1080));

    let id = f.add_client();
    let window = f.client(id).create_window();
    let surface = window.surface.clone();
    window.commit();
    f.roundtrip(id);

    let window = f.client(id).window(&surface);
    window.attach_new_buffer();
    window.set_size(240, 100);
    window.ack_last_and_commit();
    f.double_roundtrip(id);

    // Make it floating, then fullscreen.
    f.niri().layout.toggle_window_floating(None);
    f.double_roundtrip(id);

    let _ = f.client(id).window(&surface).recent_configures();
    let window_id = f.niri().layout.windows().next().unwrap().1.window.clone();

    f.niri().layout.toggle_fullscreen(&window_id);
    f.double_roundtrip(id);

    // The client keeps its own size.
    let window = f.client(id).window(&surface);
    assert_snapshot!(
        window.format_recent_configures(),
        @"size: 240 × 100, bounds: 1888 × 1048, states: [Activated, Fullscreen]"
    );

    window.ack_last_and_commit();
    f.double_roundtrip(id);

    let mapped = f.niri().layout.windows().next().unwrap().1;
    assert_ne!(mapped.visual_size(), mapped.size());
    let magnify = mapped
        .magnify_geometry()
        .expect("expected magnify geometry");
    assert_eq!(magnify.source, smithay::utils::Size::from((240., 100.)));
    assert_eq!(magnify.target, mapped.visual_size().to_f64());
}

#[test]
fn fullscreen_requested_after_initial_configure_but_before_map() {
    // Wine commits the toplevel role, lets the initial configure come back, and only then asks
    // for fullscreen -- several milliseconds later, in a separate batch. It then acks the
    // configure it already has, which is the non-fullscreen one, and maps.
    let mut f = Fixture::new();
    f.add_output(1, (1920, 1080));

    let id = f.add_client();
    let window = f.client(id).create_window();
    let surface = window.surface.clone();
    window.commit();
    f.roundtrip(id);

    // The initial configure has already gone out, without a fullscreen state.
    let window = f.client(id).window(&surface);
    assert_snapshot!(
        window.format_recent_configures(),
        @"size: 936 × 1048, bounds: 1888 × 1048, states: []"
    );

    // Now the client asks for fullscreen, then maps by acking the stale configure.
    window.set_fullscreen(None);
    window.attach_new_buffer();
    window.ack_last_and_commit();
    f.double_roundtrip(id);

    let window = f.client(id).window(&surface);
    assert_snapshot!(
        window.format_recent_configures(),
        @r"
    size: 1920 × 1080, bounds: 1888 × 1048, states: [Fullscreen]
    size: 1920 × 1080, bounds: 1888 × 1048, states: [Fullscreen, Activated]
    "
    );
}

#[test]
fn fullscreen_requested_with_buffer_less_commits_and_fixed_size_hint() {
    // Closer to what Wine actually does: buffer-less commits in between, the fullscreen request
    // repeated, and a fixed min == max size hint set just before mapping.
    let mut f = Fixture::new();
    f.add_output(1, (1920, 1080));

    let id = f.add_client();
    let window = f.client(id).create_window();
    let surface = window.surface.clone();
    window.commit();
    f.roundtrip(id);

    let window = f.client(id).window(&surface);
    window.set_fullscreen(None);
    window.commit();
    f.roundtrip(id);

    let window = f.client(id).window(&surface);
    window.set_fullscreen(None);
    window.commit();
    f.roundtrip(id);

    // Wine marks non-resizable windows as fixed size, which is also what makes niri float them.
    let window = f.client(id).window(&surface);
    window.set_min_size(1920, 1080);
    window.set_max_size(1920, 1080);
    window.attach_new_buffer();
    window.ack_last_and_commit();
    f.double_roundtrip(id);

    let window = f.client(id).window(&surface);
    assert_snapshot!(
        window.format_recent_configures(),
        @r"
    size: 936 × 1048, bounds: 1888 × 1048, states: []
    size: 1920 × 1080, bounds: 1888 × 1048, states: [Fullscreen]
    size: 1920 × 1080, bounds: 1888 × 1048, states: [Fullscreen]
    size: 1920 × 1080, bounds: 1888 × 1048, states: [Fullscreen, Activated]
    "
    );
}

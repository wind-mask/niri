use client::ClientId;
use smithay::reexports::wayland_server::protocol::wl_surface::WlSurface;
use smithay::utils::Size;
use smithay::wayland::compositor::with_states;
use smithay::wayland::fractional_scale::with_fractional_scale;
use wayland_client::protocol::wl_surface::WlSurface as ClientSurface;

use super::*;
use crate::layout::LayoutElement as _;

/// Returns the scale that niri last advertised to a surface's client.
fn preferred_scale(surface: &WlSurface) -> Option<f64> {
    with_states(surface, |states| {
        with_fractional_scale(states, |fractional| fractional.preferred_scale())
    })
}

/// Returns the mapped window's toplevel surface, as the compositor sees it.
fn window_surface(f: &mut Fixture) -> WlSurface {
    f.niri()
        .layout
        .windows()
        .next()
        .expect("expected a mapped window")
        .1
        .toplevel()
        .wl_surface()
        .clone()
}

/// Returns a config with a single window rule that sets the client scale.
fn config_with_client_scale(scale: f64) -> niri_config::Config {
    let mut config = niri_config::Config::default();
    config.window_rules.push(niri_config::WindowRule {
        upscaling: niri_config::UpscalingRule {
            client_scale: Some(niri_config::FloatOrInt(scale)),
            ..Default::default()
        },
        ..Default::default()
    });
    config
}

/// Returns a config with a single window rule that sets the virtual size.
fn config_with_virtual_size(width: i32, height: i32) -> niri_config::Config {
    let mut config = niri_config::Config::default();
    config.window_rules.push(niri_config::WindowRule {
        upscaling: niri_config::UpscalingRule {
            virtual_size: Some(niri_config::window_rule::WindowSize { width, height }),
            ..Default::default()
        },
        ..Default::default()
    });
    config
}

/// Returns the size of the last configure that the client received.
fn last_configure_size(f: &mut Fixture, id: ClientId, surface: &ClientSurface) -> (i32, i32) {
    f.client(id)
        .window(surface)
        .recent_configures()
        .last()
        .expect("expected a configure")
        .size
}

/// Makes the client apply a configured size and commit it.
fn apply_configure(
    f: &mut Fixture,
    id: ClientId,
    surface: &ClientSurface,
    width: i32,
    height: i32,
) {
    let window = f.client(id).window(surface);
    window.set_size(width as u16, height as u16);
    window.ack_last_and_commit();
    f.double_roundtrip(id);
}

/// Creates a client with a 100×100 window and lets it map, returning the client id and its
/// toplevel surface.
fn map_window(f: &mut Fixture) -> (ClientId, ClientSurface) {
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

    (id, surface)
}

/// With the `client-scale` rule, niri tells the window's client to render at the configured scale
/// instead of the output's, which is what makes the client render fewer pixels for super resolution
/// to magnify.
#[test]
fn client_scale_rule_reaches_the_client() {
    let mut f = Fixture::with_config(config_with_client_scale(0.5));
    f.add_output(1, (1920, 1080));

    map_window(&mut f);

    // The output's scale is 1, so the client would only be told 0.5 if the rule was applied.
    assert_eq!(preferred_scale(&window_surface(&mut f)), Some(0.5));
}

/// A window that no rule matches keeps using the output's scale.
#[test]
fn client_scale_rule_only_applies_to_matching_windows() {
    let mut config = niri_config::Config::default();
    config.window_rules.push(niri_config::WindowRule {
        matches: vec![niri_config::window_rule::Match {
            app_id: Some("^some-other-app$".parse().unwrap()),
            ..Default::default()
        }],
        upscaling: niri_config::UpscalingRule {
            client_scale: Some(niri_config::FloatOrInt(0.5)),
            ..Default::default()
        },
        ..Default::default()
    });

    let mut f = Fixture::with_config(config);
    f.add_output(1, (1920, 1080));

    map_window(&mut f);

    assert_eq!(preferred_scale(&window_surface(&mut f)), Some(1.));
}

/// Changing the rule sends the new scale to the client right away, even though the output's scale
/// did not change.
#[test]
fn client_scale_rule_change_is_sent_to_the_client() {
    let mut f = Fixture::with_config(config_with_client_scale(0.5));
    f.add_output(1, (1920, 1080));

    let (id, surface) = map_window(&mut f);
    assert_eq!(preferred_scale(&window_surface(&mut f)), Some(0.5));

    *f.niri().config.borrow_mut() = config_with_client_scale(2.);
    f.niri().recompute_window_rules();
    f.double_roundtrip(id);

    assert_eq!(preferred_scale(&window_surface(&mut f)), Some(2.));
    let _ = surface;
}

/// With the `virtual-size` rule, the client is asked for the configured size instead of the area
/// the layout gives the window, and the window still occupies that area, with its content magnified
/// into it.
#[test]
fn virtual_size_sets_the_client_size() {
    let mut f = Fixture::with_config(config_with_virtual_size(320, 240));
    f.add_output(1, (1920, 1080));

    let (id, surface) = map_window(&mut f);

    let (width, height) = last_configure_size(&mut f, id, &surface);
    assert_eq!((width, height), (320, 240));

    // The client applies the size it was asked for.
    apply_configure(&mut f, id, &surface, width, height);

    let mapped = f.niri().layout.windows().next().unwrap().1;
    let visual = mapped.visual_size();
    assert_ne!(
        visual,
        mapped.size(),
        "expected the window to occupy more than the size it renders at"
    );

    let magnify = mapped
        .magnify_geometry()
        .expect("expected magnify geometry");
    assert_eq!(magnify.source, Size::from((320., 240.)));
    assert_eq!(magnify.target, visual.to_f64());
}

/// Going fullscreen does not change the size the client is asked for either: it keeps rendering the
/// virtual size, and niri draws it into the whole output.
#[test]
fn virtual_size_magnify_fullscreen_keeps_the_client_size() {
    let mut f = Fixture::with_config(config_with_virtual_size(320, 240));
    f.add_output(1, (1920, 1080));

    let (id, surface) = map_window(&mut f);

    let window_id = f.niri().layout.windows().next().unwrap().1.window.clone();
    f.niri().layout.toggle_fullscreen(&window_id);
    f.double_roundtrip(id);

    let (width, height) = last_configure_size(&mut f, id, &surface);
    assert_eq!((width, height), (320, 240));

    // The client applies the size it was asked for, keeping it in fullscreen.
    apply_configure(&mut f, id, &surface, width, height);

    let mapped = f.niri().layout.windows().next().unwrap().1;
    let magnify = mapped
        .magnify_geometry()
        .expect("expected magnify geometry");
    assert_eq!(magnify.source, Size::from((320., 240.)));
    assert_eq!(magnify.target, mapped.visual_size().to_f64());
}

/// A virtual size that does not fit in the area the layout gives the window would only make the
/// window overflow it, so it is left alone.
#[test]
fn virtual_size_bigger_than_the_area_is_ignored() {
    let mut f = Fixture::with_config(config_with_virtual_size(4000, 3000));
    f.add_output(1, (1920, 1080));

    let (id, surface) = map_window(&mut f);

    assert_ne!(last_configure_size(&mut f, id, &surface), (4000, 3000));

    let mapped = f.niri().layout.windows().next().unwrap().1;
    assert_eq!(mapped.magnify_geometry(), None);
    assert_eq!(mapped.visual_size(), mapped.size());
}

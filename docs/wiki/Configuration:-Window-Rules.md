### Overview

Window rules let you adjust behavior for individual windows.
They have `match` and `exclude` directives that control which windows the rule should apply to, and a number of properties that you can set.

Window rules are processed in order of appearance in the config file.
This means that you can put more generic rules first, then override them for specific windows later.
For example:

```kdl
// Set open-maximized to true for all windows.
window-rule {
    open-maximized true
}

// Then, for Alacritty, set open-maximized back to false.
window-rule {
    match app-id="Alacritty"
    open-maximized false
}
```

> [!TIP]
> In general, you cannot "unset" a property in a later rule, only set it to a different value.
> Use the `exclude` directives to avoid applying a rule for specific windows.

Here are all matchers and properties that a window rule could have:

```kdl
window-rule {
    match title="Firefox"
    match app-id="Alacritty"
    match is-active=true
    match is-focused=false
    match is-active-in-column=true
    match is-floating=true
    match is-window-cast-target=true
    match is-urgent=true
    match at-startup=true

    // Properties that apply once upon window opening.
    default-column-width { proportion 0.75; }
    default-window-height { fixed 500; }
    open-on-output "Some Company CoolMonitor 1234"
    open-on-workspace "chat"
    open-maximized true
    open-maximized-to-edges true
    open-fullscreen true
    open-floating true
    open-focused false

    // Properties that apply continuously.
    draw-border-with-background false
    opacity 0.5
    block-minimize true
    block-out-from "screencast"
    // block-out-from "screen-capture"
    variable-refresh-rate true
    default-column-display "tabbed"
    default-floating-position x=100 y=200 relative-to="bottom-left"
    scroll-factor 0.75
    pinch-sensitivity 1.25
    on-xdg-activate "focus"

    focus-ring {
        // off
        on
        width 4
        active-color "#7fc8ff"
        inactive-color "#505050"
        urgent-color "#9b0000"
        // active-gradient from="#80c8ff" to="#bbddff" angle=45
        // inactive-gradient from="#505050" to="#808080" angle=45 relative-to="workspace-view"
        // urgent-gradient from="#800" to="#a33" angle=45
    }

    border {
        // Same as focus-ring.
    }

    shadow {
        // on
        off
        softness 40
        spread 5
        offset x=0 y=5
        draw-behind-window true
        color "#00000064"
        // inactive-color "#00000064"
    }

    tab-indicator {
        active-color "red"
        inactive-color "gray"
        urgent-color "blue"
        // active-gradient from="#80c8ff" to="#bbddff" angle=45
        // inactive-gradient from="#505050" to="#808080" angle=45 relative-to="workspace-view"
        // urgent-gradient from="#800" to="#a33" angle=45
    }

    geometry-corner-radius 12
    clip-to-geometry true
    tiled-state true
    baba-is-float true

    background-effect {
        xray true
        blur true
        noise 0.05
        saturation 3
    }

    popups {
        opacity 0.5
        geometry-corner-radius 15

        background-effect {
            xray true
            blur true
            noise 0.05
            saturation 3
        }
    }

    upscaling {
        mode "filter"
        algorithm "fsr1"
        sharpness 0.2
    }

    min-width 100
    max-width 200
    min-height 300
    max-height 300
}
```

### Window Matching

Each window rule can have several `match` and `exclude` directives.
In order for the rule to apply, a window needs to match *any* of the `match` directives, and *none* of the `exclude` directives.

```kdl
window-rule {
    // Match all Telegram windows...
    match app-id=r#"^org\.telegram\.desktop$"#

    // ...except the media viewer window.
    exclude title="^Media viewer$"

    // Properties to apply.
    open-on-output "HDMI-A-1"
}
```

Match and exclude directives have the same syntax.
There can be multiple *matchers* in one directive, then the window should match all of them for the directive to apply.

```kdl
window-rule {
    // Match Firefox windows with Gmail in title.
    match app-id="firefox" title="Gmail"
}

window-rule {
    // Match Firefox, but only when it is active...
    match app-id="firefox" is-active=true

    // ...or match Telegram...
    match app-id=r#"^org\.telegram\.desktop$"#

    // ...but don't match the Telegram media viewer.
    // If you open a tab in Firefox titled "Media viewer",
    // it will not be excluded because it doesn't match the app-id
    // of this exclude directive.
    exclude app-id=r#"^org\.telegram\.desktop$"# title="Media viewer"
}
```

Let's look at the matchers in more detail.

#### `title` and `app-id`

These are regular expressions that should match anywhere in the window title and app ID respectively.
You can read about the supported regular expression syntax [here](https://docs.rs/regex/latest/regex/#syntax).

```kdl
// Match windows with title containing "Mozilla Firefox",
// or windows with app ID containing "Alacritty".
window-rule {
    match title="Mozilla Firefox"
    match app-id="Alacritty"
}
```

Raw KDL strings can be helpful for writing out regular expressions:

```kdl
window-rule {
    exclude app-id=r#"^org\.keepassxc\.KeePassXC$"#
}
```

You can find the title and the app ID of a window by running `niri msg pick-window` and clicking on the window in question.

> [!TIP]
> Another way to find the window title and app ID is to configure the `wlr/taskbar` module in [Waybar](https://github.com/Alexays/Waybar) to include them in the tooltip:
> 
> ```json
> "wlr/taskbar": {
>     "tooltip-format": "{title} | {app_id}",
> }
> ```

#### `is-active`

Can be `true` or `false`.
Matches active windows (same windows that have the active border / focus ring color).

Every workspace on the focused monitor will have one active window.
This means that you will usually have multiple active windows (one per workspace), and when you switch between workspaces, you can see two active windows at once.

```kdl
window-rule {
    match is-active=true
}
```

#### `is-focused`

Can be `true` or `false`.
Matches the window that has the keyboard focus.

Contrary to `is-active`, there can only be a single focused window.
Also, when opening a layer-shell application launcher or pop-up menu, the keyboard focus goes to layer-shell.
While layer-shell has the keyboard focus, windows will not match this rule.

```kdl
window-rule {
    match is-focused=true
}
```

#### `is-active-in-column`

<sup>Since: 0.1.6</sup>

Can be `true` or `false`.
Matches the window that is the "active" window in its column.

Contrary to `is-active`, there is always one `is-active-in-column` window in each column.
It is the window that was last focused in the column, i.e. the one that will gain focus if this column is focused.

<sup>Since: 25.01</sup> This rule will match `true` during the initial window opening.

```kdl
window-rule {
    match is-active-in-column=true
}
```

#### `is-floating`

<sup>Since: 25.01</sup>

Can be `true` or `false`.
Matches floating windows.

> [!NOTE]
> This matcher will apply only after the window is already open.
> This means that you cannot use it to change the window opening properties like `default-window-height` or `open-on-workspace`.

```kdl
window-rule {
    match is-floating=true
}
```

#### `is-window-cast-target`

<sup>Since: 25.02</sup>

Can be `true` or `false`.
Matches `true` for windows that are target of an ongoing window screencast.

> [!NOTE]
> This only matches individual-window screencasts.
> It will not match windows that happen to be visible in a monitor screencast, for example.

```kdl
// Indicate screencasted windows with red colors.
window-rule {
    match is-window-cast-target=true

    focus-ring {
        active-color "#f38ba8"
        inactive-color "#7d0d2d"
    }

    border {
        inactive-color "#7d0d2d"
    }

    shadow {
        color "#7d0d2d70"
    }

    tab-indicator {
        active-color "#f38ba8"
        inactive-color "#7d0d2d"
    }
}
```

Example:

![A screenshot showing that only the is-window-cast-target=true windows receive the special border colors](https://github.com/user-attachments/assets/375b381e-3a87-4e94-8676-44404971d893)

#### `is-urgent`

<sup>Since: 25.05</sup>

Can be `true` or `false`.
Matches windows that request the user's attention.

```kdl
window-rule {
    match is-urgent=true
}
```

#### `at-startup`

<sup>Since: 0.1.6</sup>

Can be `true` or `false`.
Matches during the first 60 seconds after starting niri.

This is useful for properties like `open-on-output` which you may want to apply only right after starting niri.

```kdl
// Open windows on the HDMI-A-1 monitor at niri startup, but not afterwards.
window-rule {
    match at-startup=true
    open-on-output "HDMI-A-1"
}
```

### Window Opening Properties

These properties apply once, when a window first opens.

To be precise, they apply at the point when niri sends the initial configure request to the window.

#### `default-column-width`

Set the default width for the new window.

This works for floating windows too, despite the word "column" in the name.

```kdl
// Give Blender and GIMP some guaranteed width on opening.
window-rule {
    match app-id="^blender$"

    // GIMP app ID contains the version like "gimp-2.99",
    // so we only match the beginning (with ^) and not the end.
    match app-id="^gimp"

    default-column-width { fixed 1200; }
}
```

#### `default-window-height`

<sup>Since: 25.01</sup>

Set the default height for the new window.

```kdl
// Open the Firefox picture-in-picture window as floating with 480×270 size.
window-rule {
    match app-id="firefox$" title="^Picture-in-Picture$"

    open-floating true
    default-column-width { fixed 480; }
    default-window-height { fixed 270; }
}
```

#### `open-on-output`

Make the window open on a specific output.

If such an output does not exist, the window will open on the currently focused output as usual.

If the window opens on an output that is not currently focused, the window will not be automatically focused.

```kdl
// Open Firefox and Telegram (but not its Media Viewer)
// on a specific monitor.
window-rule {
    match app-id="firefox$"
    match app-id=r#"^org\.telegram\.desktop$"#
    exclude app-id=r#"^org\.telegram\.desktop$"# title="^Media viewer$"

    open-on-output "HDMI-A-1"
    // Or:
    // open-on-output "Some Company CoolMonitor 1234"
}
```

<sup>Since: 0.1.9</sup> `open-on-output` can now use monitor manufacturer, model, and serial.
Before, it could only use the connector name.

#### `open-on-workspace`

<sup>Since: 0.1.6</sup>

Make the window open on a specific [named workspace](./Configuration:-Named-Workspaces.md).

If such a workspace does not exist, the window will open on the currently focused workspace as usual.

If the window opens on an output that is not currently focused, the window will not be automatically focused.

```kdl
// Open Fractal on the "chat" workspace.
window-rule {
    match app-id=r#"^org\.gnome\.Fractal$"#

    open-on-workspace "chat"
}
```

#### `open-maximized`

Make the window open as a maximized column.

```kdl
// Maximize Firefox by default.
window-rule {
    match app-id="firefox$"

    open-maximized true
}
```

#### `open-maximized-to-edges`

<sup>Since: 25.11</sup>

Make the window open [maximized to edges](./Fullscreen-and-Maximize.md).

```kdl
window-rule {
    open-maximized-to-edges true
}
```

You can also set this to `false` to *prevent* a window from opening maximized to edges.

```kdl
window-rule {
    open-maximized-to-edges false
}
```

#### `open-fullscreen`

Make the window open [fullscreen](./Fullscreen-and-Maximize.md).

```kdl
window-rule {
    open-fullscreen true
}
```

You can also set this to `false` to *prevent* a window from opening fullscreen.

```kdl
// Make the Telegram media viewer open in windowed mode.
window-rule {
    match app-id=r#"^org\.telegram\.desktop$"# title="^Media viewer$"

    open-fullscreen false
}
```

#### `open-floating`

<sup>Since: 25.01</sup>

Make the window open in the floating layout.

```kdl
// Open the Firefox picture-in-picture window as floating.
window-rule {
    match app-id="firefox$" title="^Picture-in-Picture$"

    open-floating true
}
```

You can also set this to `false` to *prevent* a window from opening in the floating layout.

```kdl
// Open all windows in the tiling layout, overriding any auto-floating logic.
window-rule {
    open-floating false
}
```

#### `open-focused`

<sup>Since: 25.01</sup>

Set this to `false` to prevent this window from being automatically focused upon opening.

```kdl
// Don't give focus to the GIMP startup splash screen.
window-rule {
    match app-id="^gimp" title="^GIMP Startup$"

    open-focused false
}
```

You can also set this to `true` to focus the window, even if normally it wouldn't get auto-focused.

```kdl
// Always focus the KeePassXC-Browser unlock dialog.
//
// This dialog opens parented to the KeePassXC window rather than the browser,
// so it doesn't get auto-focused by default.
window-rule {
    match app-id=r#"^org\.keepassxc\.KeePassXC$"# title="^Unlock Database - KeePassXC$"

    open-focused true
}
```

### Dynamic Properties

These properties apply continuously to open windows.

#### `block-out-from`

You can block out windows from xdg-desktop-portal screencasts.
They will be replaced with solid black rectangles.

This can be useful for password managers or messenger windows, etc.
For layer-shell notification pop-ups and the like, you can use a [`block-out-from` layer rule](./Configuration:-Layer-Rules.md#block-out-from).

![Screenshot showing a window visible normally, but blocked out on OBS.](./img/block-out-from-screencast.png)

To preview and set up this rule, check the `preview-render` option in the debug section of the config.

> [!CAUTION]
> The window is **not** blocked out from third-party screenshot tools.
> If you open some screenshot tool with preview while screencasting, blocked out windows **will be visible** on the screencast.

The built-in screenshot UI is not affected by this problem though.
If you open the screenshot UI while screencasting, you will be able to select the area to screenshot while seeing all windows normally, but on a screencast the selection UI will display with windows blocked out.

```kdl
// Block out password managers from screencasts.
window-rule {
    match app-id=r#"^org\.keepassxc\.KeePassXC$"#
    match app-id=r#"^org\.gnome\.World\.Secrets$"#

    block-out-from "screencast"
}
```

Alternatively, you can block out the window out of *all* screen captures, including third-party screenshot tools.
This way you avoid accidentally showing the window on a screencast when opening a third-party screenshot preview.

This setting will still let you use the interactive built-in screenshot UI, but it will block out the window from the fully automatic screenshot actions, such as `screenshot-screen` and `screenshot-window`.
The reasoning is that with an interactive selection, you can make sure that you avoid screenshotting sensitive content.

```kdl
window-rule {
    block-out-from "screen-capture"
}
```

> [!WARNING]
> Be careful when blocking out windows based on a dynamically changing window title.
>
> For example, you might try to block out specific Firefox tabs like this:
>
> ```kdl
> window-rule {
>     // Doesn't quite work! Try to block out the Gmail tab.
>     match app-id="firefox$" title="- Gmail "
>
>     block-out-from "screencast"
> }
> ```
>
> It will work, but when switching from a sensitive tab to a regular tab, the contents of the sensitive tab **will show up on a screencast** for an instant.
>
> This is because window title (and app ID) are not double-buffered in the Wayland protocol, so they are not tied to specific window contents.
> There's no robust way for Firefox to synchronize visibly showing a different tab and changing the window title.

#### `on-xdg-activate`

<sup>Since: next release</sup>

Set what niri does when a window requests activation through [xdg-activation](https://wayland.app/protocols/xdg-activation-v1).

Values:

- `"ignore"`: do nothing: neither focus the window nor mark it urgent.
- `"set-urgent"`: always mark the window urgent instead of focusing it.
- `"focus"`: always focus the window, even for activation requests without a serial.

The default behavior, when this rule is unset, picks between focusing and marking the window urgent based on the serial that the application provides when creating the activation token.
Requests with a valid serial focus the target window, and requests without a serial mark it urgent.

Requests with a set, but invalid, serial, are always ignored, which you can change with the [`honor-xdg-activation-with-invalid-serial`](https://niri-wm.github.io/niri/Configuration%3A-Debug-Options.html#honor-xdg-activation-with-invalid-serial) debug flag.

This is useful for apps that steal focus on incoming messages or when opening popup windows like Picture-in-Picture.

```kdl
// Don't let Firefox PiP steal focus.
window-rule {
    match title="^Picture-in-Picture$"

    open-floating true
    on-xdg-activate "set-urgent"
}
```

#### `block-minimize`

Ignore `xdg_toplevel.set_minimized` requests coming from the window itself.

Minimizing the window through a key binding, through `niri msg action minimize-window`, or from a taskbar still works; only the window's own requests are ignored.

This is mainly useful for fullscreen games.
Windows games leave exclusive fullscreen and minimize themselves whenever they lose focus, which under Wine means every time you switch to another window in niri.
Without this rule, the game disappears from your layout on every window switch.

```kdl
window-rule {
    match app-id=r#"^borderlands4\.exe$"#

    block-minimize true
}
```

> [!NOTE]
> This does not stop the game from *leaving fullscreen* on focus loss, it only stops it from being taken out of the layout.
> Wine's own `WAYLANDDRV_FOCUS_LOSS=0` environment variable suppresses the focus change itself.

#### `opacity`

Set the opacity of the window.
`0.0` is fully transparent, `1.0` is fully opaque.
This is applied on top of the window's own opacity, so semitransparent windows will become even more transparent.

Opacity is applied to every surface of the window individually, so subsurfaces and pop-up menus will show window content behind them.

![Screenshot showing Adwaita Demo with a semitransparent pop-up menu.](./img/opacity-popup.png)

Also, focus ring and border with background will show through semitransparent windows (see `prefer-no-csd` and the `draw-border-with-background` window rule below).

Opacity can be toggled on or off for a window using the [`toggle-window-rule-opacity`](./Configuration:-Key-Bindings.md#toggle-window-rule-opacity) action.

```kdl
// Make inactive windows semitransparent.
window-rule {
    match is-active=false

    opacity 0.95
}
```

#### `allow-tearing`

Request screen tearing for this window, as if it had used the tearing-control protocol. Setting it
to `false` suppresses the window's own tearing requests instead.

```kdl
// Allow tearing only when osu! has keyboard focus.
window-rule {
    match app-id="^osu!$" is-focused=true

    allow-tearing true
}
```

Whether the request is actually honored depends on the output's
[`allow-tearing`](./Configuration:-Outputs.md#allow-tearing) setting, which by default only lets the
focused fullscreen window tear. If you want a windowed application to tear, set `allow-tearing true`
on the output as well.

#### `variable-refresh-rate`

<sup>Since: 0.1.9</sup>

If set to true, whenever this window displays on an output with on-demand VRR, it will enable VRR on that output.

```kdl
// Configure some output with on-demand VRR.
output "HDMI-A-1" {
    variable-refresh-rate on-demand=true
}

// Enable on-demand VRR when mpv displays on the output.
window-rule {
    match app-id="^mpv$"

    variable-refresh-rate true
}
```

<sup>Since: next release</sup> You can set `variable-refresh-rate` to `"fullscreen-only"` to activate on-demand VRR only when this window is fullscreen on this output.
The window must cover the entire screen (when niri hides the top layer-shell layer).

This mode will work even when there are overlays over the fullscreen window.

```kdl
// Enable on-demand VRR when any window is fullscreen.
window-rule {
    variable-refresh-rate "fullscreen-only"
}
```

#### `default-column-display`

<sup>Since: 25.02</sup>

Set the default display mode for columns created from this window.
Can be `normal` or `tabbed`.

This is used any time a window goes into its own column.
For example:
- Opening a new window.
- Expelling a window into its own column.
- Moving a window from the floating layout to the tiling layout.

```kdl
// Make Evince windows open as tabbed columns.
window-rule {
    match app-id="^evince$"

    default-column-display "tabbed"
}
```

#### `default-floating-position`

<sup>Since: 25.01</sup>

Set the initial position for this window when it opens on, or moves to the floating layout.

Afterward, the window will remember its last floating position.

By default, new floating windows open at the center of the screen, and windows from the tiling layout open close to their visual screen position.

The position uses logical coordinates relative to the working area.
By default, they are relative to the top-left corner of the working area, but you can change this by setting `relative-to` to one of these values: `top-left`, `top-right`, `bottom-left`, `bottom-right`, `top`, `bottom`, `left`, or `right`.

For example, if you have a bar at the top, then `x=0 y=0` will put the top-left corner of the window directly below the bar.
If instead you write `x=0 y=0 relative-to="top-right"`, then the top-right corner of the window will align with the top-right corner of the workspace, also directly below the bar.
When only one side is specified (e.g. top) the window will align to the center of that side.

The coordinates change direction based on `relative-to`.
For example, by default (top-left), `x=100 y=200` will put the window 100 pixels to the right and 200 pixels down from the top-left corner.
If you use `x=100 y=200 relative-to="bottom-left"`, it will put the window 100 pixels to the right and 200 pixels *up* from the bottom-left corner.

```kdl
// Open the Firefox picture-in-picture window at the bottom-left corner of the screen
// with a small gap.
window-rule {
    match app-id="firefox$" title="^Picture-in-Picture$"

    default-floating-position x=32 y=32 relative-to="bottom-left"
}
```

You can use single-side `relative-to` to get a dropdown-like effect.

```kdl
// Example: a "dropdown" terminal.
window-rule {
    // Match by "dropdown" app ID.
    // You need to set this app ID when running your terminal, e.g.:
    // spawn "alacritty" "--class" "dropdown"
    match app-id="^dropdown$"

    // Open it as floating.
    open-floating true
    // Anchor to the top edge of the screen.
    default-floating-position x=0 y=0 relative-to="top"
    // Half of the screen high.
    default-window-height { proportion 0.5; }
    // 80% of the screen wide.
    default-column-width { proportion 0.8; }
}
```

#### `scroll-factor`

<sup>Since: 25.02</sup>

Set a scroll factor for all scroll events sent to a window.

This will be multiplied with the scroll factor set for your input device in the [input section](./Configuration:-Input.md#pointing-devices).

```kdl
// Make scrolling in Firefox a bit slower.
window-rule {
    match app-id="firefox$"

    scroll-factor 0.75
}
```

#### `pinch-sensitivity`

<sup>Since: next release</sup>

Set the sensitivity of pinch gestures sent to a window.

This will be multiplied with the pinch sensitivity set for your touchpad in the [input section](./Configuration:-Input.md#pointing-devices).

```kdl
// Make pinch gestures in Firefox more sensitive.
window-rule {
    match app-id="firefox$"

    pinch-sensitivity 1.25
}
```

#### `draw-border-with-background`

Override whether the border and the focus ring draw with a background.

Set this to `true` to draw them as solid colored rectangles even for windows which agreed to omit their client-side decorations.
Set this to `false` to draw them as borders around the window even for windows which use client-side decorations.

This property can be useful for rectangular windows that do not support the xdg-decoration protocol.

| With Background                                  | Without Background                                  |
| ------------------------------------------------ | --------------------------------------------------- |
| ![A screenshot displaying a window with draw-border-with-background set to true](./img/simple-egl-border-with-background.png) | ![A screenshot displaying a window with draw-border-with-background set to false](./img/simple-egl-border-without-background.png) |

```kdl
window-rule {
    draw-border-with-background false
}
```

#### `focus-ring` and `border`

<sup>Since: 0.1.6</sup>

Override the focus ring and border options for the window.

These rules have the same options as the normal [`focus-ring` and `border` config in the layout section](./Configuration:-Layout.md#focus-ring-and-border), so check the documentation there.

However, in addition to `off` to disable the border/focus ring, this window rule has an `on` flag that enables the border/focus ring for the window even if it was otherwise disabled.
The `on` flag has precedence over the `off` flag, in case both are set.

```kdl
window-rule {
    focus-ring {
        off
        width 2
    }
}

window-rule {
    border {
        on
        width 8
    }
}
```

#### `shadow`

<sup>Since: 25.02</sup>

Override the shadow options for the window.

This rule has the same options as the normal [`shadow` config in the layout section](./Configuration:-Layout.md#shadow), so check the documentation there.

However, in addition to `on` to enable the shadow, this window rule has an `off` flag that disables the shadow for the window even if it was otherwise enabled.
The `on` flag has precedence over the `off` flag, in case both are set.

```kdl
// Turn on shadows for floating windows.
window-rule {
    match is-floating=true

    shadow {
        on
    }
}
```

#### `tab-indicator`

<sup>Since: 25.02</sup>

Override the tab indicator options for the window.

Options in this rule match the same options as the normal [`tab-indicator` config in the layout section](./Configuration:-Layout.md#tab-indicator), so check the documentation there.

```kdl
// Make KeePassXC tab have a dark red inactive color.
window-rule {
    match app-id=r#"^org\.keepassxc\.KeePassXC$"#

    tab-indicator {
        inactive-color "darkred"
    }
}
```

#### `geometry-corner-radius`

<sup>Since: 0.1.6</sup>

Set the corner radius of the window.

On its own, this setting will only affect the border and the focus ring—they will round their corners to match the geometry corner radius.
If you'd like to force-round the corners of the window itself, set [`clip-to-geometry true`](#clip-to-geometry) in addition to this setting.

```kdl
window-rule {
    geometry-corner-radius 12
}
```

The radius is set in logical pixels, and controls the radius of the window itself, that is, the inner radius of the border:

![A screenshot showing a window with every corner rounded](./img/geometry-corner-radius.png)

Instead of one radius, you can set four, for each corner.
The order is the same as in CSS: top-left, top-right, bottom-right, bottom-left.

```kdl
window-rule {
    geometry-corner-radius 8 8 0 0
}
```

This way, you can match GTK 3 applications which have square bottom corners:

![A screenshot showing a window with only the top corners rounded](./img/different-corner-radius.png)

#### `clip-to-geometry`

<sup>Since: 0.1.6</sup>

Clips the window to its visual geometry.

This will cut out any client-side window shadows, and also round window corners according to `geometry-corner-radius`.

![A screenshot showing a window with rounded corners, clipped to the visual geometry](./img/clip-to-geometry.png)

```kdl
window-rule {
    clip-to-geometry true
}
```

Enable border, set [`geometry-corner-radius`](#geometry-corner-radius) and `clip-to-geometry`, and you've got a classic setup:

![A screenshot showing a window with rounded corners, and a border](./img/border-radius-clip.png)

```kdl
prefer-no-csd

layout {
    focus-ring {
        off
    }

    border {
        width 2
    }
}

window-rule {
    geometry-corner-radius 12
    clip-to-geometry true
}
```

#### `tiled-state`

<sup>Since: 25.05</sup>

Informs the window that it is tiled.
Usually, windows will react by becoming rectangular and hiding their client-side shadows.
Windows that snap their size to a grid (e.g. terminals like [foot](https://codeberg.org/dnkl/foot)) will usually disable this snapping when they are tiled.

By default, niri will set the tiled state to `true` together with [`prefer-no-csd`](./Configuration:-Miscellaneous.md#prefer-no-csd) in order to improve behavior for apps that don't support server-side decorations.
You can use this window rule to override this, for example to get rectangular windows with CSD.

```kdl
// Make tiled windows rectangular while using CSD.
window-rule {
    match is-floating=false

    tiled-state true
}
```

#### `baba-is-float`

<sup>Since: 25.02</sup>

Make your windows FLOAT up and down.

This is an April Fools' 2025 feature.

```kdl
window-rule {
    match is-floating=true

    baba-is-float true
}
```

<video controls src="https://github.com/user-attachments/assets/3f4cb1a4-40b2-4766-98b7-eec014c19509">

https://github.com/user-attachments/assets/3f4cb1a4-40b2-4766-98b7-eec014c19509

</video>

#### `background-effect`

<sup>Since: 26.04</sup>

Override the background effect options for this window.

- `xray`: set to `true` to enable the xray effect, or `false` to disable it.
- `blur`: set to `true` to enable blur behind this window, or `false` to force-disable it.
- `noise`: amount of pixel noise added to the background (helps with color banding from blur).
- `saturation`: color saturation of the background (`0` is desaturated, `1` is normal, `2` is 200% saturation).

See the [window effects page](./Window-Effects.md) for an overview of background effects.

```kdl
// Make floating windows use the regular blur (if enabled),
// while tiled windows keep using the efficient xray blur.
//
// Warning: non-xray blur is currently experimental and has known limitations.
// In particular, it doesn't work during window opening and closing animations.
window-rule {
    match is-floating=true

    background-effect {
        xray false
    }
}
```

#### `popups`

<sup>Since: 26.04</sup>

Override properties for this window's pop-ups (menus and tooltips).

The properties work the same way as the corresponding window-rule properties, except that they apply to the window's pop-ups rather than to the window itself.

`opacity` is applied *on top* of the layer surface's own opacity rule, so setting both will make pop-ups more transparent than the surface.
Other properties apply independently.

> [!NOTE]
> This block affects only pop-ups created by the app via Wayland's [xdg-popup](https://wayland.app/protocols/xdg-shell#xdg_popup) (which should be most of them).
>
> Examples of things that look like pop-ups that won't work:
>
> - Fully emulated by the client, i.e. not a pop-up at all, the client just draws something that looks like a pop-up inside its window.
> These are common in game engines and in web apps, e.g. the right click menu in Google Docs or in Electron apps like Discord.
>
> - Uses a wl-subsurface instead of an xdg-popup.
> Common in older apps using GTK 3, notably Firefox still uses these for some menus.
> Subsurfaces are an indivisible part of a surface and they aren't usually pop-ups, so it wouldn't make sense for niri to apply these rules to them.
>
> These emulated pop-ups come with other downsides: they cannot reliably extend outside their window, and if the app tries to do that, they will be clipped by rules such as `clip-to-geometry`.
> So most modern apps will correctly use xdg-popup, which is the intended way to show pop-ups on Wayland.
>
> This block also does not affect input-method pop-ups, such as Fcitx.
>
> For pop-ups created by your desktop shell or desktop components, use the corresponding [layer rule](./Configuration:-Layer-Rules.md#popups).

```kdl
// Blur the background behind pop-up menus in Nautilus.
window-rule {
    match app-id="Nautilus"

    popups {
        // Matches the default libadwaita pop-up corner radius.
        geometry-corner-radius 15

        // Note: it'll look better to set background opacity
        // through your GTK theme CSS and not here.
        // This is just an example that makes it look obvious.
        opacity 0.5

        background-effect {
            blur true
        }
    }
}
```

Keep in mind that the background effect will look right only if the pop-up is shaped like a (rounded) rectangle, and the window correctly sets its Wayland geometry to exclude any shadows.
For example, GTK 4 pop-ups with pointing arrows (`has-arrow=true` property) are *not* rounded rectangles—the arrow sticks out—so if you enable blur, it will also stick out of the pop-up.

| Correct                                             | Wrong                                                                          |
|-----------------------------------------------------|--------------------------------------------------------------------------------|
| The pop-up is a rounded rectangle. Blur looks fine. | The pop-up is not a rounded rectangle. Blur extends above, where the arrow is. |
| ![](./img/popup-no-arrow.png)                       | ![](./img/popup-arrow.png)                                                     |

These pop-ups with custom shapes will need the app to implement the [ext-background-effect protocol](https://wayland.app/protocols/ext-background-effect-v1) to work properly.

#### `upscaling`

<sup>Since: next release</sup>

Render this window with super resolution: instead of the compositor's built-in bilinear sampling, niri resamples the window's buffer with an algorithm of your choice.

This is meant for windows that render at a lower resolution than the area they are displayed in, like a game or a visual novel running in an upscaling mode, or an app that does not support fractional scaling. If the client buffer is already the same size as its display area, there is nothing to resample and the effect does not do anything useful.

- `mode`: what to do with the window's content.
  - `"filter"`: only replace the sampling used to magnify the buffer. The window keeps its display geometry, so clicking and input are unaffected. This is the default.
  - `"off"`: turn super resolution off (useful to override an earlier rule).
  - `"magnify"`: draw the window's content scaled into a larger display area, without resizing the client. This is what you want for a game that runs at a lower resolution than the screen: put the window in fullscreen (with the usual `fullscreen-window` action or an `open-fullscreen` rule) and niri fills the whole output with its content instead, keeping the aspect ratio and leaving black bars where needed. The client keeps its own size, so leaving fullscreen needs no restoring on its side, and input is mapped back into the client's coordinates, including the letterbox bars, which do not take input. With [`virtual-size`](#upscaling), the same thing happens wherever the window is: the client always renders the configured size, and niri scales it into whatever area the layout gives the window.
- `algorithm`: resampling algorithm to use.
  - `"bilinear"`: explicit bilinear sampling. This is what the compositor does anyway, so it is mostly useful as a reference.

  - `"bicubic"`: Catmull-Rom bicubic sampling. A good default for sharp content.
  - `"fsr1"`: AMD [FidelityFX Super Resolution 1](https://github.com/GPUOpen-Effects/FidelityFX-FSR), that is, EASU followed by RCAS. Good for content with edges and gradients, and it can sharpen.
  - `"cunny2"`: a [CuNNy](https://github.com/funnyplanter/CuNNy) network, which reconstructs the image instead of resampling it, trained on illustrations. It is the one to try for visual novels, anime-style art and text, where its learned reconstruction keeps lines and letters cleaner than the resampling kernels do. It is also the most expensive algorithm here: the network runs at the client buffer's resolution, so unlike `"fsr1"` its cost follows the client's resolution rather than the output's, and it needs a few source-sized textures while it runs. Unlike the other algorithms, it does not sharpen; the details it recovers come from the network.
  - `"cunny2-large"`: the larger CuNNy network, `CuNNy-3x12-NVL`, with one more convolution than `"cunny2"`. Noticeably more expensive, and worth trying where the extra detail shows, like text-heavy illustrations.
  - `"anime4k"`: not implemented yet; it falls back to `"bicubic"` and says so in the log.
- `sharpness`: sharpening strength for algorithms that sharpen, which is `"fsr1"`. Like FSR1's own setting, it is in stops of halving: `0` is maximum sharpening, `1` is half of that, and `2` is a quarter. Defaults to `0.2`.

- `client-scale`: the scale niri advertises to this window's client, for making it render at a different pixel density than the output's scale. Clients that handle fractional scaling render one buffer pixel per `client-scale` logical pixels, so a value below the output's scale makes the client render fewer pixels than its display area has, which is exactly what this effect magnifies. This is how you turn a client that would render at the output's full resolution into a low-resolution one on purpose: with `scale 1.5` on the output, `client-scale 1.0` makes the client render one pixel per logical pixel, and niri magnifies it 1.5× while drawing it, with the configured algorithm. It combines with `mode "magnify"`, where the client keeps its own logical size while rendering fewer pixels. Keep in mind that the client cannot tell this apart from a smaller output scale: everything it draws itself, including its decorations, pop-ups and text, is sized for the configured scale too, so it will look smaller than the rest of the desktop. Zero means the output's scale, which is the default. This does not enable the effect by itself.

- `virtual-size`: the size to tell this window's client to render at, as two arguments, the width and the height in logical pixels, like `virtual-size 1280 720`. The client is configured with this size instead of the area the layout gives the window, and niri draws its content scaled into that area, keeping the aspect ratio and leaving black bars where needed. This is the other half of "render less and let niri magnify": `client-scale` decides how many pixels the client renders per logical pixel, and this decides the logical size it renders at. Setting it turns on the `"magnify"` mode, since that is the only mode where it does anything, but you can set the mode explicitly as well. The size is only used when it fits in the area the layout gives the window: one that is bigger than the area in both dimensions would only make the window overflow it, so such a window is left alone, and its client is given the area's size like any other window.

Super resolution stays off unless a rule enables it, either by setting `mode` to something other than `"off"`, or by setting `algorithm` or `virtual-size` on its own.

```kdl
// Resample a visual novel that renders at a lower resolution as sharply as possible.
window-rule {
    match app-id="^vn4ever$"

    upscaling {
        mode "filter"
        algorithm "fsr1"
        sharpness 0.2
    }
}
```

```kdl
// Make a game render 1280×720, and magnify it to the output with FSR1.
window-rule {
    match app-id="^steam_app_default$"

    upscaling {
        algorithm "fsr1"
        virtual-size 1280 720
        client-scale 1.0
    }
}
```

> [!NOTE]
> The effect has to actually run, so while it is on, niri cannot hand the window's buffer to the display directly. This rules out [direct scanout](./Configuration:-Debug-Options.md#disable-direct-scanout) for that window, which costs performance. Enabling it for a window that is not magnified, like a fullscreen game at its native resolution, gains nothing and still costs that.

With `mode "magnify"`, only a window that is actually fullscreen is magnified, unless it has a `virtual-size`: that makes niri scale the client's content into the area the layout gives the window wherever the window is. A window that is not fullscreen and has no virtual size is unaffected. Pop-ups are drawn scaled along with the window, but with bilinear sampling rather than with the configured algorithm.

Whether there is anything to magnify depends on the client. One that renders at the physical resolution of the output (which is what Wayland clients that handle fractional scaling do) leaves no detail for the effect to recover, and such a window is drawn the normal way, without running the effect at all. One that keeps rendering at a lower resolution than its display area, like an app that predates fractional scaling on a scaled output, does. The `client-scale` and `virtual-size` options are how you make a client render at a lower resolution on purpose.

> [!NOTE]
> [`clip-to-geometry`](#clip-to-geometry) and `geometry-corner-radius` are not applied to resampled windows yet. Pop-ups, borders and the cursor are composited normally.

> [!NOTE]
> `"cunny2"` reads the client buffer the way the sampling shaders do, so a buffer whose orientation it cannot read, and a renderer where one of the network's passes cannot be compiled or run, both fall back to `"bicubic"`. The log says so.

> [!TIP]
> niri logs what it does with this effect, which is the quickest way to find out why it does not seem to apply. Run niri with `RUST_LOG=niri=info,niri::render_helpers::superres=debug` to see which algorithm each window ends up with, whether the buffer turned out to be the same size as the display area, and when a window gets magnified. For detailed per-pass GPU and buffer diagnostics without flooding the system journal, set `NIRI_GPU_LOG=/tmp/niri/gpu_log` (optionally with `NIRI_GPU_LOG_FILTER` and `NIRI_GPU_LOG_MAX_BYTES`).
>
> The offscreen passes (`"fsr1"`'s EASU and `"cunny2"` / `"cunny2-large"`'s neural network) only run when the window is inside the output's visible viewport and its pixels actually change: windows scrolled off-screen skip the offscreen passes entirely, and clients that keep submitting identical frames (such as games redrawing a static scene at 60 fps) reuse the cached result via a compact 32×32 GPU source digest whenever the pass workload is above the measured threshold.
>
> The same logs are where the passes are timed once per second, which is what makes the algorithms comparable on your own GPU: `super resolution: window 5 fsr1, 2560×1440: easu 0.84 ms in 60 passes + draw 1.12 ms in 60 passes, 118 ms of GPU time in 1000 ms`. `easu` is the offscreen pass that runs whenever the client draws a new frame, `draw` is the final draw, which is where the single-pass algorithms and RCAS run, `network` is a run of a neural network when the algorithm is `"cunny2"` or `"cunny2-large"`, and the last number is what the effect costs in total as a share of the GPU time available in that second. Drivers without timer query support (the capability line in the log says `timer = no`) report no numbers.

#### Size Overrides

You can amend the window's minimum and maximum size in logical pixels.

Keep in mind that the window itself always has a final say in its size.
These values instruct niri to never ask the window to be smaller than the minimum you set, or to be bigger than the maximum you set.

> [!NOTE]
> `max-height` will only apply to automatically-sized windows if it is equal to `min-height`.
> Either set it equal to `min-height`, or change the window height manually after opening it with `set-window-height`.
>
> This is a limitation of niri's window height distribution algorithm.

```kdl
window-rule {
    min-width 100
    max-width 200
    min-height 300
    max-height 300
}
```

```kdl
// Fix OBS with server-side decorations missing a minimum width.
window-rule {
    match app-id=r#"^com\.obsproject\.Studio$"#

    min-width 876
}
```

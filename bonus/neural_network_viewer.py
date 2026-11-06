#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "my_torch" / "src"))
import pygame

import numpy as np

from my_torch import NeuralNetwork


WINDOW_SIZE = (1280, 800)
SIDEBAR_WIDTH = 260
MAX_VISIBLE_NEURONS = 12

BACKGROUND = (21, 25, 34)
PANEL = (30, 36, 48)
PANEL_LIGHT = (39, 47, 62)
TEXT = (232, 237, 245)
MUTED = (151, 164, 184)
ACCENT = (94, 170, 255)
POSITIVE = (79, 190, 140)
NEGATIVE = (242, 105, 117)


def main() -> int:
    pygame.init()
    screen = pygame.display.set_mode(WINDOW_SIZE, pygame.RESIZABLE)
    pygame.display.set_caption("MyTorch - Network viewer")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 22)
    small_font = pygame.font.Font(None, 18)
    title_font = pygame.font.Font(None, 30)

    models_dir = PROJECT_ROOT / "my_torch" / "trained_models"
    model_paths: list[Path] = []
    selected_path: Path | None = None
    network: NeuralNetwork | None = None
    error_message: str | None = None
    list_scroll = 0
    selected_neuron: tuple[int, int] | None = None
    node_hitboxes: list[tuple[pygame.Rect, int, int]] = []
    model_rows: list[tuple[pygame.Rect, Path]] = []
    refresh_rect = pygame.Rect(16, 68, SIDEBAR_WIDTH - 32, 38)

    def load_selected_model(path: Path | None) -> None:
        nonlocal selected_path, network, error_message, selected_neuron
        selected_path = path
        selected_neuron = None
        network = None
        error_message = None
        if path is None:
            return
        try:
            network = NeuralNetwork.load(str(path))
            if not isinstance(network, NeuralNetwork):
                raise ValueError("File does not contain a neural network")
        except Exception as error:
            error_message = f"{type(error).__name__}: {error}"

    def refresh_models() -> None:
        nonlocal model_paths, list_scroll
        current_name = selected_path.name if selected_path else None
        model_paths = sorted(
            (path for path in models_dir.glob("*.nn") if path.is_file()),
            key=lambda path: path.name.lower(),
        ) if models_dir.is_dir() else []
        list_scroll = min(list_scroll, max(0, len(model_paths) - 1))
        retained = next(
            (path for path in model_paths if path.name == current_name),
            model_paths[0] if model_paths else None,
        )
        if retained != selected_path:
            load_selected_model(retained)
        elif retained is not None and network is None and error_message is None:
            load_selected_model(retained)

    def draw_text(
        surface: pygame.Surface,
        text: str,
        x: int,
        y: int,
        color: tuple[int, int, int] = TEXT,
        use_font: pygame.font.Font = font,
    ) -> None:
        surface.blit(use_font.render(text, True, color), (x, y))

    def draw_diagram(surface: pygame.Surface, area: pygame.Rect) -> None:
        nonlocal node_hitboxes
        node_hitboxes = []
        if network is None:
            message = error_message or "Select a model to display."
            draw_text(surface, message[:100], area.x + 20, area.y + 28, NEGATIVE)
            return

        layers = [layer for layer in network._layers]
        layer_sizes = [int(layers[0].weights.shape[0])]
        layer_sizes.extend(int(layer.weights.shape[1]) for layer in layers)
        visible_counts = [min(size, MAX_VISIBLE_NEURONS) for size in layer_sizes]
        x_gap = area.width / max(1, len(layer_sizes))
        positions: list[list[tuple[int, int]]] = []

        for layer_index, (size, visible_count) in enumerate(
            zip(layer_sizes, visible_counts)
        ):
            x = round(area.x + x_gap * (layer_index + 0.5))
            top = area.y + 72
            bottom = area.bottom - 42
            y_gap = (bottom - top) / max(1, visible_count - 1)
            nodes = [
                (x, round(top + node_index * y_gap))
                for node_index in range(visible_count)
            ]
            positions.append(nodes)

        max_weight = max(
            (float(np.max(np.abs(layer.weights))) for layer in layers),
            default=1.0,
        ) or 1.0
        for layer_index, layer in enumerate(layers):
            weights = layer.weights
            for source_index, (start_x, start_y) in enumerate(positions[layer_index]):
                for target_index, (end_x, end_y) in enumerate(
                    positions[layer_index + 1]
                ):
                    weight = float(weights[source_index, target_index])
                    color = POSITIVE if weight >= 0 else NEGATIVE
                    strength = min(abs(weight) / max_weight, 1.0)
                    width = 1 + round(6 * strength**0.5)
                    pygame.draw.line(
                        surface, color, (start_x, start_y), (end_x, end_y), width
                    )

        for layer_index, (size, nodes) in enumerate(zip(layer_sizes, positions)):
            draw_text(
                surface,
                f"Layer {layer_index}",
                nodes[0][0] - 33,
                area.y + 25,
                MUTED,
                small_font,
            )
            for node_index, (x, y) in enumerate(nodes):
                selected = selected_neuron == (layer_index, node_index)
                color = ACCENT if selected else PANEL_LIGHT
                pygame.draw.circle(surface, color, (x, y), 13)
                pygame.draw.circle(surface, TEXT, (x, y), 13, 1)
                node_hitboxes.append(
                    (pygame.Rect(x - 15, y - 15, 30, 30), layer_index, node_index)
                )
            if size > len(nodes):
                draw_text(
                    surface,
                    f"+ {size - len(nodes)} more",
                    nodes[0][0] - 28,
                    area.bottom - 28,
                    MUTED,
                    small_font,
                )
            draw_text(
                surface,
                f"{size} neurons",
                nodes[0][0] - 37,
                area.bottom - 8,
                MUTED,
                small_font,
            )

    def draw_details(surface: pygame.Surface, area: pygame.Rect) -> None:
        if network is None:
            return
        weights_layers = network._layers
        if selected_neuron is None:
            draw_text(
                surface,
                "Click a neuron to inspect its weights and bias.",
                area.x + 20,
                area.y + 18,
                MUTED,
            )
            return

        layer_index, neuron_index = selected_neuron
        if layer_index < len(weights_layers):
            layer = weights_layers[layer_index]
            row = np.asarray(layer.weights[neuron_index], dtype=float)
            bias = (
                float(weights_layers[layer_index - 1].biases[neuron_index])
                if layer_index > 0
                else None
            )
            relation = f"weights to layer {layer_index + 1}"
        else:
            layer = weights_layers[layer_index - 1]
            row = np.asarray(layer.weights[:, neuron_index], dtype=float)
            bias = float(layer.biases[neuron_index])
            relation = f"incoming weights from layer {layer_index - 1}"

        draw_text(
            surface,
            f"Neuron {neuron_index} in layer {layer_index} — {relation}",
            area.x + 20,
            area.y + 16,
        )
        bias_text = f"{bias:+.6f}" if bias is not None else "n/a (input)"
        draw_text(
            surface,
            f"Bias: {bias_text}    weights: {len(row)}    "
            f"min: {float(np.min(row)):+.4f}    max: {float(np.max(row)):+.4f}    "
            f"mean: {np.mean(row):+.4f}",
            area.x + 20,
            area.y + 48,
            MUTED,
            small_font,
        )
        largest = np.argsort(np.abs(row))[-8:][::-1]
        summary = "   ".join(
            f"n{int(index)}: {float(row[index]):+.5f}" for index in largest
        )
        draw_text(
            surface,
            f"Largest weights: {summary}",
            area.x + 20,
            area.y + 78,
            ACCENT,
            small_font,
        )

    def draw() -> None:
        width, height = screen.get_size()
        screen.fill(BACKGROUND)
        pygame.draw.rect(screen, PANEL, (0, 0, SIDEBAR_WIDTH, height))
        draw_text(screen, "Trained models", 16, 22, use_font=title_font)
        pygame.draw.rect(screen, PANEL_LIGHT, refresh_rect, border_radius=6)
        draw_text(screen, "Refresh model list", refresh_rect.x + 12, refresh_rect.y + 10)

        model_rows.clear()
        row_height = 34
        list_top = 124
        visible_rows = max(0, (height - list_top - 16) // row_height)
        for row_index, path in enumerate(
            model_paths[list_scroll : list_scroll + visible_rows]
        ):
            rect = pygame.Rect(
                12, list_top + row_index * row_height, SIDEBAR_WIDTH - 24, 30
            )
            if path == selected_path:
                pygame.draw.rect(screen, PANEL_LIGHT, rect, border_radius=5)
            draw_text(screen, path.name[:28], rect.x + 10, rect.y + 7)
            model_rows.append((rect, path))
        if not model_paths:
            draw_text(screen, "No .nn files found.", 16, list_top, MUTED, small_font)

        content_x = SIDEBAR_WIDTH + 24
        content_width = width - content_x - 24
        draw_text(screen, "Neural network", content_x, 20, use_font=title_font)
        if selected_path is not None:
            draw_text(screen, selected_path.name, content_x, 55, ACCENT)
        graph_area = pygame.Rect(
            content_x, 92, max(200, content_width), max(280, height - 315)
        )
        pygame.draw.rect(screen, PANEL, graph_area, border_radius=8)
        draw_diagram(screen, graph_area)

        details_area = pygame.Rect(content_x, graph_area.bottom + 14, max(200, content_width), 180)
        pygame.draw.rect(screen, PANEL, details_area, border_radius=8)
        draw_details(screen, details_area)
        draw_text(
            screen,
            "Thicker = stronger weight   Green = positive   Red = negative",
            content_x,
            height - 24,
            MUTED,
            small_font,
        )
        pygame.display.flip()

    refresh_models()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEWHEEL:
                list_scroll = max(
                    0, min(list_scroll - event.y, max(0, len(model_paths) - 1))
                )
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if refresh_rect.collidepoint(event.pos):
                    refresh_models()
                    continue
                for rect, path in model_rows:
                    if rect.collidepoint(event.pos):
                        if path != selected_path:
                            load_selected_model(path)
                        break
                else:
                    for rect, layer_index, neuron_index in node_hitboxes:
                        if rect.collidepoint(event.pos):
                            selected_neuron = (layer_index, neuron_index)
                            break
        draw()
        clock.tick(30)

    pygame.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
Visualização 3D do Sistema Pick-and-Place

Este script gera visualizações 3D interativas e animações da trajetória
do sistema pick-and-place, diferenciando cada ciclo por cores.

Uso:
    python pick_and_place_visualizer.py
"""

import pandas as pd
import numpy as np

try:
    import plotly.graph_objects as go
    import plotly.express as px
except ImportError:
    print("Plotly não está instalado. Instale com:")
    print("    pip install plotly>=5.18.0")
    exit(1)

# =============================================================================
# CONFIGURAÇÕES
# =============================================================================

# Caminho para o arquivo de dados
DATA_PATH = "downloads/data/cenario1_with_ffill.csv"

# Configurações de animação (em milissegundos)
ANIMATION_FRAME_DURATION = 100  # Duração de cada frame
ANIMATION_TRANSITION_DURATION = 50  # Duração da transição entre frames

# Paleta de cores para os ciclos
COLOR_PALETTE = px.colors.qualitative.Set1


# =============================================================================
# FUNÇÕES DE CARREGAMENTO E PROCESSAMENTO
# =============================================================================

def load_and_detect_cycles(path: str) -> pd.DataFrame:
    """
    Carrega o CSV e detecta os ciclos baseado na coluna JOB_DONE.
    
    Um ciclo é identificado quando JOB_DONE transita de 0 para 1.
    
    Args:
        path: Caminho para o arquivo CSV
        
    Returns:
        DataFrame com coluna adicional 'cycle_id'
    """
    # Carregar dados
    df = pd.read_csv(path)
    
    # Converter timestamp para datetime se existir
    if 'SourceTimeStamp' in df.columns:
        df['SourceTimeStamp'] = pd.to_datetime(df['SourceTimeStamp'])
    
    # Detectar transições de JOB_DONE (0 -> 1)
    # Isso marca o FIM de cada ciclo
    job_done_transitions = (df['JOB_DONE'].diff() == 1).astype(int)
    
    # Criar cycle_id: incrementa a cada transição
    # O primeiro ciclo é 0, depois incrementa a cada JOB_DONE
    df['cycle_id'] = job_done_transitions.cumsum()
    
    # Ajustar para que o ciclo comece em 1
    df['cycle_id'] = df['cycle_id'] + 1
    
    # Adicionar índice temporal para animação
    df['frame_idx'] = range(len(df))
    
    print(f"Dados carregados: {len(df)} pontos")
    print(f"Ciclos detectados: {df['cycle_id'].nunique()}")
    
    return df


# =============================================================================
# FUNÇÕES DE VISUALIZAÇÃO
# =============================================================================

def plot_3d_static(df: pd.DataFrame) -> None:
    """
    Cria visualização 3D estática da trajetória com cores por ciclo.
    
    Args:
        df: DataFrame com colunas x_pos, y_pos, z_pos e cycle_id
    """
    fig = go.Figure()
    
    # Obter ciclos únicos
    cycles = df['cycle_id'].unique()
    n_colors = len(COLOR_PALETTE)
    
    for i, cycle in enumerate(cycles):
        cycle_data = df[df['cycle_id'] == cycle]
        color = COLOR_PALETTE[i % n_colors]
        
        # Adicionar linha da trajetória
        fig.add_trace(go.Scatter3d(
            x=cycle_data['x_pos'],
            y=cycle_data['y_pos'],
            z=cycle_data['z_pos'],
            mode='lines+markers',
            name=f'Ciclo {cycle}',
            line=dict(color=color, width=3),
            marker=dict(size=3, color=color),
            hovertemplate=(
                '<b>Ciclo %d</b><br>' % cycle +
                'X: %{x:.2f}<br>' +
                'Y: %{y:.2f}<br>' +
                'Z: %{z:.2f}<br>' +
                '<extra></extra>'
            )
        ))
        
        # Marcar ponto inicial do ciclo
        fig.add_trace(go.Scatter3d(
            x=[cycle_data['x_pos'].iloc[0]],
            y=[cycle_data['y_pos'].iloc[0]],
            z=[cycle_data['z_pos'].iloc[0]],
            mode='markers',
            name=f'Início Ciclo {cycle}',
            marker=dict(size=8, color=color, symbol='diamond'),
            showlegend=False,
            hovertemplate=f'<b>Início Ciclo {cycle}</b><extra></extra>'
        ))
        
        # Marcar ponto final do ciclo
        fig.add_trace(go.Scatter3d(
            x=[cycle_data['x_pos'].iloc[-1]],
            y=[cycle_data['y_pos'].iloc[-1]],
            z=[cycle_data['z_pos'].iloc[-1]],
            mode='markers',
            name=f'Fim Ciclo {cycle}',
            marker=dict(size=8, color=color, symbol='square'),
            showlegend=False,
            hovertemplate=f'<b>Fim Ciclo {cycle}</b><extra></extra>'
        ))
    
    # Configurar layout
    fig.update_layout(
        title=dict(
            text='<b>Trajetória 3D do Pick-and-Place</b>',
            x=0.5,
            font=dict(size=20)
        ),
        scene=dict(
            xaxis_title='Posição X',
            yaxis_title='Posição Y',
            zaxis_title='Posição Z',
            xaxis=dict(range=[0, 10]),
            yaxis=dict(range=[0, 10]),
            zaxis=dict(range=[0, 10]),
            aspectmode='cube'
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        margin=dict(l=0, r=0, t=50, b=0)
    )
    
    print("\nAbrindo visualização 3D estática no navegador...")
    fig.show()


def plot_3d_animated(df: pd.DataFrame) -> None:
    """
    Cria animação 3D da trajetória com cores por ciclo.
    
    Args:
        df: DataFrame com colunas x_pos, y_pos, z_pos, cycle_id e frame_idx
    """
    # Criar dados acumulativos para animação
    # Cada frame mostra todos os pontos até aquele momento
    frames = []
    
    # Obter ciclos únicos para coloração
    cycles = df['cycle_id'].unique()
    n_colors = len(COLOR_PALETTE)
    cycle_colors = {cycle: COLOR_PALETTE[i % n_colors] for i, cycle in enumerate(cycles)}
    
    # Criar um frame para cada N pontos (para não ficar muito lento)
    step = max(1, len(df) // 100)  # Máximo de ~100 frames
    frame_indices = list(range(0, len(df), step)) + [len(df) - 1]
    
    for idx in frame_indices:
        frame_data = df.iloc[:idx + 1]
        traces = []
        
        for cycle in frame_data['cycle_id'].unique():
            cycle_data = frame_data[frame_data['cycle_id'] == cycle]
            color = cycle_colors[cycle]
            
            traces.append(go.Scatter3d(
                x=cycle_data['x_pos'],
                y=cycle_data['y_pos'],
                z=cycle_data['z_pos'],
                mode='lines+markers',
                name=f'Ciclo {cycle}',
                line=dict(color=color, width=3),
                marker=dict(size=3, color=color)
            ))
        
        # Adicionar marcador da posição atual
        current_point = df.iloc[idx]
        traces.append(go.Scatter3d(
            x=[current_point['x_pos']],
            y=[current_point['y_pos']],
            z=[current_point['z_pos']],
            mode='markers',
            name='Posição Atual',
            marker=dict(size=12, color='red', symbol='circle'),
            showlegend=True
        ))
        
        frames.append(go.Frame(
            data=traces,
            name=str(idx)
        ))
    
    # Criar figura inicial (primeiro frame)
    initial_traces = frames[0].data if frames else []
    
    fig = go.Figure(
        data=initial_traces,
        frames=frames
    )
    
    # Adicionar controles de animação
    fig.update_layout(
        title=dict(
            text='<b>Animação 3D do Pick-and-Place</b>',
            x=0.5,
            font=dict(size=20)
        ),
        scene=dict(
            xaxis_title='Posição X',
            yaxis_title='Posição Y',
            zaxis_title='Posição Z',
            xaxis=dict(range=[0, 10]),
            yaxis=dict(range=[0, 10]),
            zaxis=dict(range=[0, 10]),
            aspectmode='cube'
        ),
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                y=0.1,
                x=0.1,
                xanchor="left",
                buttons=[
                    dict(
                        label="▶ Play",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(
                                    duration=ANIMATION_FRAME_DURATION,
                                    redraw=True
                                ),
                                transition=dict(
                                    duration=ANIMATION_TRANSITION_DURATION
                                ),
                                fromcurrent=True,
                                mode="immediate"
                            )
                        ]
                    ),
                    dict(
                        label="⏸ Pause",
                        method="animate",
                        args=[
                            [None],
                            dict(
                                frame=dict(duration=0, redraw=False),
                                transition=dict(duration=0),
                                mode="immediate"
                            )
                        ]
                    )
                ]
            )
        ],
        sliders=[
            dict(
                active=0,
                yanchor="top",
                xanchor="left",
                currentvalue=dict(
                    font=dict(size=12),
                    prefix="Frame: ",
                    visible=True,
                    xanchor="right"
                ),
                transition=dict(duration=ANIMATION_TRANSITION_DURATION),
                pad=dict(b=10, t=50),
                len=0.9,
                x=0.1,
                y=0,
                steps=[
                    dict(
                        args=[
                            [str(frame_indices[i])],
                            dict(
                                frame=dict(duration=ANIMATION_FRAME_DURATION, redraw=True),
                                transition=dict(duration=ANIMATION_TRANSITION_DURATION),
                                mode="immediate"
                            )
                        ],
                        label=str(frame_indices[i]),
                        method="animate"
                    )
                    for i in range(len(frame_indices))
                ]
            )
        ],
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        margin=dict(l=0, r=0, t=50, b=100)
    )
    
    print("\nAbrindo animação 3D no navegador...")
    print(f"  - Duração do frame: {ANIMATION_FRAME_DURATION}ms")
    print(f"  - Duração da transição: {ANIMATION_TRANSITION_DURATION}ms")
    print(f"  - Total de frames: {len(frames)}")
    fig.show()


def plot_3d_trajectory_with_speed(df: pd.DataFrame) -> None:
    """
    Cria visualização 3D com velocidade média representada por cores.
    
    Args:
        df: DataFrame com colunas x_pos, y_pos, z_pos e AVG_SPEED
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter3d(
        x=df['x_pos'],
        y=df['y_pos'],
        z=df['z_pos'],
        mode='lines+markers',
        marker=dict(
            size=4,
            color=df['AVG_SPEED'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title='Velocidade Média')
        ),
        line=dict(
            color=df['AVG_SPEED'],
            colorscale='Viridis',
            width=3
        ),
        hovertemplate=(
            'X: %{x:.2f}<br>' +
            'Y: %{y:.2f}<br>' +
            'Z: %{z:.2f}<br>' +
            'Velocidade: %{marker.color:.2f}<br>' +
            '<extra></extra>'
        )
    ))
    
    fig.update_layout(
        title=dict(
            text='<b>Trajetória 3D - Colorida por Velocidade</b>',
            x=0.5,
            font=dict(size=20)
        ),
        scene=dict(
            xaxis_title='Posição X',
            yaxis_title='Posição Y',
            zaxis_title='Posição Z',
            xaxis=dict(range=[0, 10]),
            yaxis=dict(range=[0, 10]),
            zaxis=dict(range=[0, 10]),
            aspectmode='cube'
        ),
        margin=dict(l=0, r=0, t=50, b=0)
    )
    
    print("\nAbrindo visualização 3D com velocidade no navegador...")
    fig.show()


# =============================================================================
# MENU PRINCIPAL
# =============================================================================

def main():
    """Menu principal para seleção do tipo de visualização."""
    print("=" * 60)
    print("  VISUALIZAÇÃO 3D DO SISTEMA PICK-AND-PLACE")
    print("=" * 60)
    print()
    
    # Carregar dados
    print(f"Carregando dados de: {DATA_PATH}")
    df = load_and_detect_cycles(DATA_PATH)
    print()
    
    while True:
        print("-" * 40)
        print("Opções de Visualização:")
        print("-" * 40)
        print("  [1] Trajetória 3D estática (cores por ciclo)")
        print("  [2] Animação 3D (progressão temporal)")
        print("  [3] Trajetória 3D (cores por velocidade)")
        print("  [0] Sair")
        print("-" * 40)
        
        choice = input("Escolha uma opção: ").strip()
        
        if choice == "1":
            plot_3d_static(df)
        elif choice == "2":
            plot_3d_animated(df)
        elif choice == "3":
            plot_3d_trajectory_with_speed(df)
        elif choice == "0":
            print("\nEncerrando...")
            break
        else:
            print("\nOpção inválida. Tente novamente.")
        
        print()


if __name__ == "__main__":
    main()

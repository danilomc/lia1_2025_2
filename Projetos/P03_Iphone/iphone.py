import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings

# --- Configuração da Página e Título ---
st.set_page_config(page_title="Sistema de Análise e Previsão de Séries Temporais", layout="wide")
st.title("Análise e Previsão de Interesse (Google Trends)")
st.write("Esta aplicação analisa uma série temporal de dados do Google Trends, decompõe seus componentes e prevê valores futuros.")

# Ignorar warnings para uma interface mais limpa
warnings.filterwarnings("ignore")

# --- Barra Lateral (Input) ---
with st.sidebar:
    st.header("Configurações da Análise")
    uploaded_file = st.file_uploader("Escolha o arquivo CSV do Google Trends:", type=['csv'])
    
    forecast_periods = st.slider(
        "Selecione o número de semanas para prever:",
        min_value=4,
        max_value=52,
        value=12,
        step=4
    )

    st.subheader("Parâmetros do Modelo SARIMA")
    st.info("Ajuste as ordens do modelo para encontrar o melhor ajuste.")
    # Parâmetros não sazonais (p,d,q)
    p = st.slider('Ordem Autoregressiva (p)', 0, 5, 1, key='p')
    d = st.slider('Ordem de Diferenciação (d)', 0, 2, 1, key='d')
    q = st.slider('Ordem de Média Móvel (q)', 0, 5, 1, key='q')

    # Parâmetros sazonais (P,D,Q,m)
    P = st.slider('Ordem Autoregressiva Sazonal (P)', 0, 2, 1, key='P')
    D = st.slider('Ordem de Diferenciação Sazonal (D)', 0, 2, 1, key='D')
    Q = st.slider('Ordem de Média Móvel Sazonal (Q)', 0, 2, 1, key='Q')
    # m = 52 (semanal com sazonalidade anual) é fixo

    analyze_button = st.button("Iniciar Análise e Previsão")

# --- Funções Auxiliares ---
def load_and_prepare_data(file):
    """
    Carrega e prepara os dados do Google Trends a partir de um arquivo CSV.
    """
    try:
        df = pd.read_csv(file, skiprows=2)
        df.columns = ['Date', 'Interest']
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df['Interest'] = pd.to_numeric(df['Interest'].replace('<1', '0'), errors='coerce').fillna(0).astype(int)
        df = df.asfreq('W-SUN', fill_value=df['Interest'].mean())
        return df
    except Exception as e:
        st.error(f"Erro ao processar o arquivo CSV: {e}")
        return None

# --- Área Principal de Exibição ---
if uploaded_file and analyze_button:
    try:
        with st.spinner("Processando os dados e treinando o modelo... Por favor, aguarde."):
            
            data = load_and_prepare_data(uploaded_file)
            
            if data is not None:
                st.header("1. Visualização da Série Temporal Original")
                st.write("Este gráfico mostra o interesse no tópico ao longo do tempo, de acordo com os dados fornecidos.")
                
                fig, ax = plt.subplots(figsize=(12, 6))
                sns.lineplot(data=data, x=data.index, y='Interest', ax=ax, color='dodgerblue')
                ax.set_title('Interesse ao Longo do Tempo', fontsize=16)
                ax.set_xlabel('Data')
                ax.set_ylabel('Nível de Interesse')
                ax.grid(True, linestyle='--', alpha=0.6)
                st.pyplot(fig)

                # --- Decomposição da Série Temporal ---
                st.header("2. Decomposição da Série Temporal")
                st.write("""
                A decomposição nos ajuda a entender a estrutura da série temporal, separando-a em três componentes:
                - **Tendência:** A direção geral dos dados.
                - **Sazonalidade:** Padrões que se repetem em intervalos fixos.
                - **Resíduos:** A parte aleatória dos dados.
                """)
                
                decomposition = seasonal_decompose(data['Interest'], model='additive', period=52)
                
                fig_decomp = plt.figure(figsize=(12, 8))
                
                ax1 = fig_decomp.add_subplot(311)
                decomposition.trend.plot(ax=ax1, color='dodgerblue'); ax1.set_title('Tendência'); ax1.set_xlabel('')
                
                ax2 = fig_decomp.add_subplot(312)
                decomposition.seasonal.plot(ax=ax2, color='dodgerblue'); ax2.set_title('Sazonalidade'); ax2.set_xlabel('')

                ax3 = fig_decomp.add_subplot(313)
                decomposition.resid.plot(ax=ax3, color='dodgerblue'); ax3.set_title('Resíduos')
                
                plt.tight_layout()
                st.pyplot(fig_decomp)

                # --- Previsão com o Modelo SARIMA ---
                st.header("3. Previsão de Interesse para as Próximas Semanas")
                st.write(f"Utilizando um modelo SARIMA com os parâmetros definidos para prever o interesse nas próximas **{forecast_periods} semanas**.")
                
                try:
                    # Definir e treinar o modelo SARIMA com os parâmetros da barra lateral
                    order_params = (p, d, q)
                    seasonal_order_params = (P, D, Q, 52)

                    model = SARIMAX(data['Interest'],
                                    order=order_params,
                                    seasonal_order=seasonal_order_params,
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)

                    results = model.fit(disp=False)

                    # Fazer as previsões
                    forecast_object = results.get_forecast(steps=forecast_periods)
                    predictions = forecast_object.predicted_mean
                    conf_int_df = forecast_object.conf_int()
                    conf_int_df.columns = ['Limite Inferior', 'Limite Superior']

                    # Criar um índice de datas para as previsões
                    future_index = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=forecast_periods, freq='W-SUN')
                    
                    forecast_df = pd.DataFrame(predictions.values, index=future_index, columns=['Previsão'])
                    conf_int_df.index = future_index
                    
                    # Plotar os resultados
                    fig_forecast, ax_forecast = plt.subplots(figsize=(12, 6))
                    sns.lineplot(data=data, x=data.index, y='Interest', ax=ax_forecast, label='Dados Históricos', color='black')
                    sns.lineplot(data=forecast_df, x=forecast_df.index, y='Previsão', ax=ax_forecast, label='Previsão', color='red', linestyle='--')
                    ax_forecast.fill_between(conf_int_df.index,
                                             conf_int_df['Limite Inferior'],
                                             conf_int_df['Limite Superior'],
                                             color='red', alpha=0.2, label='Intervalo de Confiança (95%)')
                    
                    ax_forecast.set_title(f'Previsão de Interesse para as Próximas {forecast_periods} Semanas', fontsize=16)
                    ax_forecast.set_xlabel('Data')
                    ax_forecast.set_ylabel('Nível de Interesse')
                    ax_forecast.legend()
                    ax_forecast.grid(True, linestyle='--', alpha=0.6)
                    st.pyplot(fig_forecast)
                    
                    st.subheader("Valores Previstos")
                    display_df = forecast_df.copy()
                    display_df['Limite Inferior'] = conf_int_df['Limite Inferior']
                    display_df['Limite Superior'] = conf_int_df['Limite Superior']
                    display_df.index = display_df.index.strftime('%Y-%m-%d')
                    st.dataframe(display_df.round(2))

                except Exception as model_error:
                    st.error(f"Erro ao treinar o modelo SARIMA: {model_error}")
                    st.warning("Verifique se os parâmetros (p, d, q, P, D, Q) são válidos. Combinações inadequadas podem causar erros.")

                st.success("Análise e Previsão concluídas com sucesso!")

    except Exception as e:
        st.error(f"Ocorreu um erro inesperado durante a análise: {e}")

elif not uploaded_file and analyze_button:
    st.warning("Por favor, carregue um arquivo CSV para iniciar a análise.")

else:
    st.info("Aguardando o upload de um arquivo CSV e o comando para iniciar a análise.")


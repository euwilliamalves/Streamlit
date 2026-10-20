
import re
import os
import sqlglot
import streamlit as st

from pathlib import Path
from sqlglot import exp
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.sql_database.query import (
    create_sql_query_chain
)

# 1. Configuração da página
st.set_page_config(
    page_title="CHAT SQL",
    page_icon="💬",
    layout="centered"
)

st.title("💬 CHAT SQL")
st.caption("""O Sakila é um banco de dados de exemplo criado para estudar SQL e bancos relacionais.
 Ele representa o funcionamento de uma locadora de filmes, com informações sobre filmes, atores, clientes, aluguéis, pagamentos e lojas.
 
        Tabelas: Atores - films - clientes - aluguéis - pagamentos - categorias""")

# 2. Configuração
load_dotenv()
key = os.getenv("OPENAI_KEY")

if not key:
    st.error("A variável OPENAI_KEY não foi encontrada no arquivo .env.")
    st.stop()

pasta = Path(__file__).resolve().parent
caminho = pasta / "sakila-data.db"

if not caminho.exists():
    st.error(f"Banco de dados não encontrado: {caminho}")
    st.stop()

# 3. Conexão com o banco
@st.cache_resource
def conectar_banco(caminho_db):
    return SQLDatabase.from_uri(
        f"sqlite:///{caminho_db}"
    )

banco = conectar_banco(caminho.as_posix())

if not banco.get_usable_table_names():
    st.error("Nenhuma tabela encontrada no banco SQLite.")
    st.stop()

# 4. Modelo
@st.cache_resource
def configurar_llm(api_key):
    return ChatOpenAI(model = "gpt-3.5-turbo", temperature = 0.5, api_key = api_key)

llm = configurar_llm(key)

# 5. Cadeia SQL
@st.cache_resource
def configurar_cadeia(_llm, _banco):
    return create_sql_query_chain(
        llm=_llm,
        db=_banco,
        k=10
    )

cadeia_sql = configurar_cadeia(llm, banco)

prompt = """
## Objetivo
Você é um agente especializado em consultas SQL. Sua função é interpretar as perguntas do usuário,
consultar o banco de dados e apresentar respostas baseadas exclusivamente nos resultados reais retornados pelo SQL.

## Tabelas disponíveis
- Atores
- Filmes
- Clientes
- Aluguéis
- Pagamentos
- Categorias

### REGRA OBRIGATÓRIA: EXIBIR OS DADOS RETORNADOS PELO SQL
Quando o usuário solicitar uma lista, relação ou consulta de registros,
você DEVE apresentar os dados reais retornados pelo banco de dados. Nunca substitua os resultados por uma descrição do que a consulta retornou.

**Regras:**

1. Execute a consulta SQL e leia todas as linhas retornadas.
2. Exiba todos os registros retornados, sem omitir, resumir ou inventar linhas.
3. Não responda apenas com frases como:
   - "A consulta retornou uma lista de atores."
   - "Foram encontrados 10 registros."
   - "A lista contém os campos actor_id, first_name e last_name."
4. Faça uma abordangem antes de responder a pergunta.
5. Caso te peça o total de alguma tabela, utilize o ( SELECT * FROM tabela WHERE coluna )
6. Se a consulta utilizar `LIMIT 10`, apresente os 10 registros retornados, caso existam.
7. Se nenhum registro for retornado, informe que nenhum registro foi encontrado.
8. Se houver erro na execução SQL, informe que não foi possível obter os dados. Nunca invente resultados.
9. Preserve os valores exatamente como retornados pelo banco de dados.
10. Não mostre o código SQL na resposta final, a menos que o usuário solicite.

**Exemplo de comportamento esperado**

Pergunta do usuário: "Traga uma lista de atores".

Se o banco retornar os campos `ID`, `PRIMEIRO NOME` e `SOBRENOME`, a resposta deverá seguir este formato, preenchido exclusivamente com os dados reais retornados pelo SQL:

| ID | PRIMEIRO NOME | SOBRENOME |
|---|---|---|
| Valor retornado | Nome retornado | Sobrenome retornado |

A tabela acima ilustra apenas o formato. Substitua os valores ilustrativos pelos dados reais da consulta. Nunca apresente esses textos ilustrativos como se fossem registros verdadeiros.
**Verificação final:** antes de responder, confirme que a tabela apresentada contém os valores reais de cada linha retornada pelo banco, e não apenas uma descrição dos resultados.

Regra obrigatória para interpretar resultados SQL de contagem
Interprete o resultado real da consulta antes de formular a resposta. Nunca conclua que uma tabela está vazia sem verificar o que a consulta retornou.
Diferencie zero registros de uma consulta que contou zero valores não nulos.
COUNT(*) conta todas as linhas da tabela.
COUNT(coluna) conta somente os valores não nulos dessa coluna.
Se COUNT(actor_id) retornar 0, informe que a contagem dessa coluna foi zero. Não afirme automaticamente que não existem atores na tabela.
Quando o usuário perguntar quantos registros existem em uma tabela inteira, prefira COUNT(*). Exemplo:

SELECT COUNT(*) AS total_atores
FROM actor;

Valide a consulta antes de executá-la. Se o objetivo for contar todas as linhas, não utilize COUNT(coluna) quando COUNT(*) for a opção adequada.
Interprete corretamente o resultado:
Se COUNT(*) retornar [(200,)], responda: “A tabela possui 200 registros.”
Se COUNT(*) retornar [(0,)], responda: “A tabela não possui registros.”
Se a consulta falhar, informe que não foi possível obter a contagem. Não invente resultados.
Não confunda o resultado de uma consulta com uma conclusão mais ampla. Afirmações como “a tabela está vazia” só podem ser feitas quando os dados retornados pela consulta sustentarem essa conclusão.
Não altere os dados nem invente resultados. Utilize exclusivamente o resultado real do banco de dados.
"""



# 6. Prompt para resposta final
prompt_resposta = ChatPromptTemplate.from_messages([
    (
        "system", prompt
    ),
    (
        "human",
        """
        Pergunta: {input}
        Consulta SQL: {sql}
        Resultado do banco: {resultados}

        Formule uma resposta objetiva.
        """
    )
])

# 7. Validação do SQL
def validar_sql(sql: str):
    sql = re.sub(
        r"```(?:sql)?\s*|```",
        "",
        sql,
        flags=re.IGNORECASE
    ).strip()

    sql = re.sub(
        r"^SQLQuery:\s*",
        "",
        sql,
        flags=re.IGNORECASE
    ).strip()

    consultas = sqlglot.parse(sql, read="sqlite")

    if len(consultas) != 1 or consultas[0] is None:
        raise ValueError("Somente uma consulta é permitida.")

    consulta = consultas[0]

    if not isinstance(
        consulta,
        (exp.Select, exp.Union, exp.Intersect, exp.Except)
    ):
        raise ValueError("Apenas consultas SELECT são permitidas.")

    proibidos = (
        exp.Insert,
        exp.Update,
        exp.Delete,
        exp.Drop,
        exp.Create,
        exp.Alter,
    )

    if any(
        isinstance(no, proibidos)
        for no in consulta.walk()
    ):
        raise ValueError("Operação SQL não permitida.")

    return sql


# 8. Processamento da pergunta
def responder(pergunta: str):
    gerado = cadeia_sql.invoke({
        "question": pergunta
    })

    sql_validado = validar_sql(gerado)

    resultados = banco.run(sql_validado)

    resposta = (prompt_resposta | llm).invoke({
        "input": pergunta,
        "sql": sql_validado,
        "resultados": resultados
    })

    return {
        "sql": sql_validado,
        "resultados": resultados,
        "resposta": resposta.content
    }


# 9. Histórico do chat
if "mensagens" not in st.session_state:
    st.session_state.mensagens = []

for mensagem in st.session_state.mensagens:
    with st.chat_message(mensagem["role"]):
        st.markdown(mensagem["content"])

        if mensagem["role"] == "assistant" and mensagem.get("sql"):
            with st.expander("Ver consulta SQL"):
                st.code(mensagem["sql"], language="sql")

                st.markdown("**Resultado do banco**")
                st.code(
                    mensagem["resultados"],
                    language="text"
                )

# 10. Entrada do usuário
pergunta = st.chat_input(
    "Pergunte sobre os dados do banco..."
)

if pergunta:
    st.session_state.mensagens.append({
        "role": "user",
        "content": pergunta
    })

    with st.chat_message("user"):
        st.markdown(pergunta)

    with st.chat_message("assistant"):
        with st.spinner("Consultando o banco..."):
            try:
                resultado = responder(pergunta)

                st.markdown(resultado["resposta"])

                with st.expander("Ver consulta SQL"):
                    st.code(resultado["sql"], language="sql")

                    st.markdown("**Resultado do banco**")
                    st.code(
                        resultado["resultados"],
                        language="text"
                    )

                st.session_state.mensagens.append({
                    "role": "assistant",
                    "content": resultado["resposta"],
                    "sql": resultado["sql"],
                    "resultados": resultado["resultados"]
                })

            except Exception as erro:
                mensagem_erro = (
                    f"Não foi possível processar a pergunta: {erro}"
                )
                st.error(mensagem_erro)

                st.session_state.mensagens.append({
                    "role": "assistant",
                    "content": mensagem_erro
                })

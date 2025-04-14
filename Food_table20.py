import pandas as pd
import dash
from dash import dcc, html, Input, Output, State, ctx, dash_table
import plotly.express as px

# Load and clean your data
df = pd.read_excel('sorted2.xlsx')
df2 = pd.read_excel("environment.xlsx")

env_columns = ['food_emissions_land_use',
       'food_emissions_farm', 'food_emissions_animal_feed',
       'food_emissions_processing', 'food_emissions_transport',
       'food_emissions_retail', 'food_emissions_packaging',
       'food_emissions_losses']

# Fix all numeric columns that may contain commas instead of dots
exclude = ['id', 'FOODNAME', 'FUCLASS', 'recs']
numeric_columns = [col for col in df.columns if col not in exclude]

for col in numeric_columns:
    df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
    df[col] = pd.to_numeric(df[col], errors='coerce')

nutrients = ['energy (kJ)', 'fat (g)', 'carbohydrate (g)', 'protein (g)']
vitamin_columns = ['thiamin (vitamin B1) (mg)', 'vitamin A (µg)', 'carotenoids (µg)',
       'vitamin B-12 (cobalamin) (µg)', 'vitamin C (ascorbic acid) (mg)',
       'vitamin D (µg)', 'vitamin E  (mg)', 'vitamin K (µg)']

for col in ['CO2/100g'] + nutrients + vitamin_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df_clean = df.dropna(subset=['CO2/100g'] + nutrients)

# Initialize Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)

app.layout = html.Div([
    dcc.Tabs([
        dcc.Tab(label='Scatter & Food Details', children=[
            html.H1("Food Nutrition vs. CO2 Emissions"),

            html.Div([
                html.P("Select the nutrient you are interested in from the dropdown menu. The higher the nutrient value of the food, the higher it appears in the plot. High carbon dioxide emissions move the food to the right."),
                html.P("Click the FUCLASS legend items to filter food categories. Click a food dot in the plot to see its nutrition details below."),
            ], style={"padding": "10px","backgroundColor": "#f9f9f9","border": "1px solid #ccc","borderRadius": "10px","marginBottom": "20px"}),


            html.Label("Select Y-Axis Nutrient:"),
            dcc.Dropdown(
                id='y-axis-dropdown',
                options=[{'label': col, 'value': col} for col in nutrients],
                value='protein (g)'
            ),

            dcc.Graph(id='scatter-plot'),
            html.Div(id='food-details'),
        ]),
		
	    dcc.Tab(label='Environmental Effects', children=[
            html.H2("Environmental Impact of Foods"),
            html.Label("Select an Environmental Effect:"),
            dcc.Dropdown(
                id='emission-dropdown',
                options=[{'label': v, 'value': v} for v in env_columns],
                value=env_columns[0]
            ),
            dcc.Graph(id='effect-plot')
        ]),

        dcc.Tab(label='Meal Planner', children=[
            html.H2("Meal Planner"),

            html.Label("Select Food Category (recs):"),
            dcc.Dropdown(
                id='meal-recs-dropdown',
                options=[{'label': c, 'value': c} for c in sorted(df_clean['recs'].dropna().unique())],
                value=None,
				placeholder="Choose a food category"
            ),

            html.Label("Select Food:"),
            dcc.Dropdown(id='meal-food-dropdown', placeholder="Choose a food category"),

            html.Label("Enter quantity in grams:"),
            dcc.Input(id='food-quantity', name = 'food-quantity', type='number', value=100),

            html.Button('Add to Meal', id='add-button', n_clicks=0),

            dash_table.DataTable(
                id='meal-table',
                columns=[
                   {"name": "Food", "id": "Food"},
                   {"name": "Quantity (g)", "id": "Quantity"},
                   {"name": "Energy (kJ)", "id": "Energy"},
                   {"name": "CO₂ (g)", "id": "CO2"},
                   {"name": "", "id": "Delete", "presentation": "markdown"}
                ],
               data=[],
                style_table={'marginTop': '20px'},
                style_cell={'textAlign': 'left'},
                markdown_options={"html": True}
            ),
				
			dcc.Store(id='meal-storage', data=[]),
            html.Div(id='totals-output'),            
            dcc.Graph(id="meal-pie-chart")  # <- Put graph directly, no need for extra html.Div
        ]),

        dcc.Tab(label='Top Vitamin Foods', children=[
            html.H2("Top 15 Foods Rich in Selected Vitamin"),
            html.Label("Select a Vitamin:"),
            dcc.Dropdown(
                id='vitamin-dropdown',
                options=[{'label': v, 'value': v} for v in vitamin_columns],
                value=vitamin_columns[0]
            ),
            dcc.Graph(id='top-vitamin-plot')
        ])

    ])
])


@app.callback(
    Output('scatter-plot', 'figure'),
    Input('y-axis-dropdown', 'value')
)
def update_scatter(selected_nutrient):
    df_plot = df_clean[['CO2/100g', selected_nutrient, 'FOODNAME', 'FUCLASS']].dropna()
    fig = px.scatter(
        df_plot,
        x='CO2/100g',
        y=selected_nutrient,
        hover_name='FOODNAME',
        color='FUCLASS',
        labels={'CO2/100g': 'CO₂ Emissions (g/100g)', selected_nutrient: selected_nutrient},
        title=f'{selected_nutrient} vs. CO2 Emissions'
    )
    fig.update_layout(legend_itemclick="toggleothers", legend_itemdoubleclick="toggle")
    return fig
	
@app.callback(
    Output('food-details', 'children'),
    Input('scatter-plot', 'clickData')
)
def update_food_details(click_data):
    if not click_data:
        return html.P("Click on a food item in the plot to see details.")
    
    food_name = click_data['points'][0]['hovertext']
    row = df_clean[df_clean['FOODNAME'] == food_name]
    if row.empty:
        return html.P("Food not found.")
    
    record = row.iloc[0].to_dict()
    return html.Div([
        html.H4(f"Details for {food_name}"),
        html.Ul([html.Li(f"{k}: {v}") for k, v in record.items()])
    ])
	
@app.callback(
    Output('effect-plot', 'figure'),
    Input('emission-dropdown', 'value') 
)
def update_environmental_charts(selected_effect):
    env_df = df2[['Entity'] + env_columns].dropna()
    fig = px.bar(
        env_df.sort_values(by=selected_effect, ascending=False).head(20),
        x='Entity',
        y=selected_effect,
        color='Entity',
        title=f"Top 20 Foods by {selected_effect}",
        template='plotly_white'  # <-- This is the fix
    )
    fig.update_layout(xaxis_tickangle=-45)
    return fig

@app.callback(
    Output('meal-food-dropdown', 'options'),
    Input('meal-recs-dropdown', 'value')
)
def update_food_dropdown(selected_recs):
    if selected_recs is None:
        return []
    filtered_df = df_clean[df_clean['recs'] == selected_recs]
    return [{'label': food, 'value': food} for food in sorted(filtered_df['FOODNAME'].unique())]

@app.callback(
    Output('meal-storage', 'data'),
    Output('meal-table', 'data'),
    Output('totals-output', 'children'),
    Output('meal-table', 'active_cell'),
    Output('meal-table', 'selected_cells'),
    Output('meal-pie-chart', 'figure'),
    Input('add-button', 'n_clicks'),
    Input('meal-table', 'active_cell'),
    State('meal-food-dropdown', 'value'),
    State('food-quantity', 'value'),
    State('meal-storage', 'data'),
    prevent_initial_call=True
)
def update_meal(n_clicks, active_cell, selected_food, quantity, current_data):
    return handle_meal_update(n_clicks, active_cell, selected_food, quantity, current_data)

def handle_meal_update(n_clicks, active_cell, selected_food, quantity, current_data):
    triggered_id = ctx.triggered_id
    current_data = current_data or []

    # Deleting a row: if the active cell is in the "Delete" column,
    # remove that row from current_data.
    if triggered_id == 'meal-table' and active_cell and active_cell.get('column_id') == 'Delete':
        idx = active_cell['row']
        if 0 <= idx < len(current_data):
            current_data.pop(idx)

    # Adding a food: if the Add button was clicked and a food and quantity are selected,
    # then create a new entry and append it to current_data.
    elif triggered_id == 'add-button' and selected_food and quantity:
        # Get the row of data for the selected food
        row = df_clean[df_clean['FOODNAME'] == selected_food].iloc[0]
        energy = row['energy (kJ)'] * quantity / 100
        co2 = row['CO2/100g'] * quantity / 100
        protein = row['protein (g)'] * quantity / 100  # <--- Define it here
        recs = row['recs']

    new_entry = {
        "Food": selected_food,
        "Quantity": quantity,
        "Energy": round(energy, 2),
        "CO2": round(co2, 2),
        "protein (g)": round(protein, 2),  # <--- Store it
        "Delete": '[🗑️](#)'
    }
    current_data.append(new_entry)



    # Ensure that every row in the data has a Delete cell.
    for row in current_data:
        row["Delete"] = '[🗑️](#)'

    # Calculate totals
    total_energy = round(sum(item['Energy'] for item in current_data), 2)
    total_co2 = round(sum(item['CO2'] for item in current_data), 2)
    total_protein = round(sum(item['protein (g)'] for item in current_data), 2)
	
    totals_div = html.Div([
        html.H4("Total for Meal"),
        html.P(f"Total Energy: {total_energy} kJ"),
        html.P(f"Total CO2: {total_co2} g"),
		html.P(f"Total Protein: {total_protein} g")
    ])	

    protein = row['protein (g)'] * quantity / 100
	
	# Create the pie chart using sunburst to allow color by recs
    df_meal = pd.DataFrame(current_data)
    if not df_meal.empty:
        df_meal_grouped = df_meal.groupby(['Food'], as_index=False).first()
        df_meal_grouped['recs'] = df_meal_grouped['Food'].map(
            df_clean.set_index('FOODNAME')['recs'].to_dict()
        )

        fig = px.sunburst(
            df_meal_grouped,
            path=['recs', 'Food'],
            values='Quantity',
            color='recs',
            hover_data={'Quantity': True},
            title='Meal Composition by Food Category and Item'
        )
    else:
        fig = px.pie(
            names=['None'],
            values=[1],
            title='Meal Composition by Food'
        )



    return (
        current_data,  # Updated meal-storage.data
        current_data,  # Updated meal-table.data
        totals_div,
        None, [], fig
    )


@app.callback(
    Output('top-vitamin-plot', 'figure'),
    Input('vitamin-dropdown', 'value')
)
def top_vitamin_plot(vitamin):
    df_vit = df_clean[['FOODNAME', vitamin, 'FUCLASS', 'CO2/100g']].dropna()
    df_vit = df_vit.sort_values(by=vitamin, ascending=False).head(15)
    food_order = df_vit['FOODNAME'].tolist()

    fig = px.bar(
        df_vit,
        x='FOODNAME',
        y=vitamin,
        color='FUCLASS',
        text=df_vit['CO2/100g'].round(2).astype(str) + " g CO₂",
        category_orders={'FOODNAME': food_order}
    )

    fig.update_layout(
        title_text=f"Top 15 Foods Rich in {vitamin}",
        title_x=0.5,
        showlegend=True,
        xaxis_title=None,
        yaxis_title=None,
        xaxis_tickangle=-45,
        margin=dict(t=100, b=120)
    )

    fig.update_traces(
        textposition='outside',
        cliponaxis=False
    )

    return fig
	

	




if __name__ == '__main__':
    app.run(debug=True)

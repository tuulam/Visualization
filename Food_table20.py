import pandas as pd
import dash
from dash import dcc, html, Input, Output, State, ctx, dash_table
import plotly.express as px
from dash import MATCH, ALL

# Load and clean your data
df = pd.read_excel('sorted3.xlsx')
df2 = pd.read_excel("environment.xlsx")

carbon_threshold = 50

env_columns = ['Food emissions of land use',
       'Food emissions of farms', 'Food emissions of animal feed',
       'Food emissions of processing', 'Food emissions of transport',
       'Food emissions of retail', 'Food emissions of packaging',
       'Food emissions of losses']
	   
# Recommended Daily Levels (example values, adjust as needed)
recommended_daily_levels = {
    'vitamin C, mg': '90 mg',
    'vitamin A, µg': '900 µg',
    'vitamin D, µg': '20 µg',
	'vitamin E, µg': '4 µg',
	'vitamin K, µg': '120 µg',
    'vitamin B12, µg': '2.4 µg',
	'thiamin, mg': '1.2 mg',
    'folate, µg': '400 µg',
    'iron, mg': '18 mg',
	'carotenoids, mg': '2 mg',
    'calcium, mg': '1300 mg',
    'magnesium, mg': '420 mg'
}


# Fix all numeric columns that may contain commas instead of dots
exclude = ['id', 'FOODNAME', 'FOODCLASS', 'recs']
numeric_columns = [col for col in df.columns if col not in exclude]

for col in numeric_columns:
    df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
    df[col] = pd.to_numeric(df[col], errors='coerce')

nutrients = ['energy (kJ)', 'energy (kCal)', 'fat (g)', 'carbohydrate (g)', 'protein (g)']
vitamin_columns = ['thiamin, mg', 'vitamin A, µg', 'carotenoids, mg',
       'vitamin B12, µg', 'vitamin C, mg', 'vitamin D, µg', 'vitamin E, µg', 'vitamin K, µg']

for col in ['CO2/100g'] + nutrients + vitamin_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df_clean = df.dropna(subset=['CO2/100g'] + nutrients)

def find_alternatives(food_row, df, co2_threshold=carbon_threshold, tolerance=5):
    df_other = df[df['FOODNAME'] != food_row['FOODNAME']]
    df_low_co2 = df_other[df_other['CO2/100g'] < co2_threshold]

    protein = food_row['protein (g)']
    calories = food_row['energy (kCal)']

    alternatives = df_low_co2[
        (df_low_co2['protein (g)'].between(protein - tolerance, protein + tolerance)) &
        (df_low_co2['energy (kCal)'].between(calories - 50, calories + 50))
    ]

    return alternatives[['FOODNAME', 'protein (g)', 'energy (kCal)', 'CO2/100g']]

# Initialize Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)


app.layout = html.Div([
    dcc.Tabs([
        dcc.Tab(label='Scatter & Food Details', children=[
            html.H1("Food Nutrition vs. CO2 Emissions"),

            html.Div([
                html.P("Select the nutrient you are interested in from the dropdown menu. The higher the nutrient value of the food, the higher it appears in the plot. High carbon dioxide emissions move the food to the right."),
                html.P("Click the FOODCLASS legend items to filter food categories. Click a food dot in the plot to see its nutrition details below."),
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
			
			html.Div(id='sustainability-recommendation'),
			html.Div(id='sustainability-warning'),



            dash_table.DataTable(
                id='meal-table',
                columns=[
                   {"name": "Food", "id": "Food"},
                   {"name": "Quantity (g)", "id": "Quantity"},
				   {"name": "Energy (kJ)", "id": "Energy"},
                   {"name": "Energy (kCal)", "id": "Calories"},
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
            dcc.Graph(id="meal-pie-chart"),

            html.Div([
                html.P("It is recommended to eat at least 500 g of vegetables, fruit, berries, and mushrooms. Of this amount, half should consist of berries and fruit, and the rest vegetables."),
                html.P("It is recommended to eat fish two to three times a week, using a variety of different species in turn.Your weekly intake of meat products and red meat should not exceed 500 g. One portion of fish or meat, when cooked, weighs some 100–150 g."),
				html.P("You can have 30 g of nuts and seeds a day. Legumes are recommended at 50–100 g per day."),
				html.P("The recommended daily intake of cereal products, which include cooked whole grain pasta, barley or rice, or some other whole grain side dish, or a slice of bread, is 600 ml for women and 900 ml for men. At least half of this amount should be whole grain cereals."),
				html.P("It is not recommended to increase the current consumption of poultry meat for environmental reasons. At most, one egg per day can be part of a health-promoting diet.")
            ], style={"padding": "10px","backgroundColor": "#f9f9f9","border": "1px solid #ccc","borderRadius": "10px","marginBottom": "20px"}),

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
    df_plot = df_clean[['CO2/100g', selected_nutrient, 'FOODNAME', 'FOODCLASS']].dropna()
    fig = px.scatter(
        df_plot,
        x='CO2/100g',
        y=selected_nutrient,
        hover_name='FOODNAME',
        color='FOODCLASS',
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
        env_df.sort_values(by=selected_effect, ascending=False).round(2).head(20),
        x='Entity',
        y=selected_effect,
        color='Entity',
        title=f"Top 20 Foods by {selected_effect}",
        template='plotly_white'  
    )
    fig.update_layout(
	    xaxis_tickangle=-45, 
        xaxis_title=None,
        yaxis_title=None)
    return fig

@app.callback(
    Output('meal-storage', 'data'),
    Output('meal-table', 'data'),
    Output('totals-output', 'children'),
    Output('meal-table', 'active_cell'),
    Output('meal-table', 'selected_cells'),
    Output('meal-pie-chart', 'figure'),
    Output('sustainability-recommendation', 'children'),
    Output('meal-food-dropdown', 'value'),
    Output('food-quantity', 'value'),
    Output('meal-food-dropdown', 'options'),
    Input('add-button', 'n_clicks'),
    Input('meal-table', 'active_cell'),
    Input({'type': 'alt-button', 'index': ALL}, 'n_clicks'),
    Input('meal-recs-dropdown', 'value'),
    State('meal-food-dropdown', 'value'),
    State('food-quantity', 'value'),
    State('meal-storage', 'data'),
    prevent_initial_call=True
)
def unified_meal_callback(n_clicks_add, active_cell, alt_clicks, selected_recs, selected_food, quantity, current_data):
    triggered_id = ctx.triggered_id
    current_data = current_data or []
    recommendation_div = html.Div()
    set_value = None

    # 1. Handle deletion (if clicking the delete button)
    if triggered_id == 'meal-table' and active_cell and active_cell.get('column_id') == 'Delete':
        idx = active_cell['row']
        if 0 <= idx < len(current_data):
            current_data.pop(idx)
        recommendation_div = html.Div()

    # 2. Handle clicking an alternative button (replace food)
    elif isinstance(triggered_id, dict) and triggered_id.get('type') == 'alt-button':
        # Find the index of the original food item in the table
        selected_food = triggered_id['index']  # the alternative food selected
        quantity = 100  # default quantity (you can make this dynamic if needed)

        # Find the original food entry in the meal storage and remove it
        food_idx = next((i for i, item in enumerate(current_data) if item['Food'] == selected_food), None)
        if food_idx is not None:
            current_data.pop(food_idx)  # Remove the original food from the table

        # Add the alternative food to the meal storage
        row = df_clean[df_clean['FOODNAME'] == selected_food].iloc[0]
        energy = row['energy (kJ)'] * quantity / 100
        calories = row['energy (kCal)'] * quantity / 100
        co2 = row['CO2/100g'] * quantity / 100
        protein = row['protein (g)'] * quantity / 100

        current_data.append({
            "Food": selected_food,
            "Quantity": quantity,
            "Energy": round(energy, 2),
            "Calories": round(calories, 2),
            "CO2": round(co2, 2),
            "protein (g)": round(protein, 2),
            "Delete": '[🗑️](#)'
        })

        # Check for sustainable alternatives again after replacing
        if row['CO2/100g'] > carbon_threshold:
            alternatives = find_alternatives(row, df_clean)
            if not alternatives.empty:
                recommendation_div = html.Div([  # Provide alternatives as buttons
                    html.P("🌍 This item has high CO₂ emissions. Consider these alternatives:"),
                    html.Div([
                        html.Button(
                            f"{alt['FOODNAME']} (CO₂: {alt['CO2/100g']}g, Protein: {alt['protein (g)']}g)",
                            id={'type': 'alt-button', 'index': alt['FOODNAME']},
                            n_clicks=0,
                            style={'margin': '3px'}
                        )
                        for _, alt in alternatives.head(3).iterrows()
                    ])
                ])
            else:
                recommendation_div = html.P("🌍 This item has high CO₂ emissions, but no good alternative was found.")
        else:
            recommendation_div = html.Div()

    # 3. Handle add button to add food to table
    elif triggered_id == 'add-button' and selected_food and quantity:
        row = df_clean[df_clean['FOODNAME'] == selected_food].iloc[0]
        energy = row['energy (kJ)'] * quantity / 100
        calories = row['energy (kCal)'] * quantity / 100
        co2 = row['CO2/100g'] * quantity / 100
        protein = row['protein (g)'] * quantity / 100

        current_data.append({
            "Food": selected_food,
            "Quantity": quantity,
            "Energy": round(energy, 2),
            "Calories": round(calories, 2),
            "CO2": round(co2, 2),
            "protein (g)": round(protein, 2),
            "Delete": '[🗑️](#)'
        })

        # Check for sustainable alternatives
        if row['CO2/100g'] > 2.5:
            alternatives = find_alternatives(row, df_clean)
            if not alternatives.empty:
                recommendation_div = html.Div([  # Provide alternatives as buttons
                    html.P("🌍 This item has high CO₂ emissions. Consider these alternatives:"),
                    html.Div([
                        html.Button(
                            f"{alt['FOODNAME']} (CO₂: {alt['CO2/100g']}g, Protein: {alt['protein (g)']}g)",
                            id={'type': 'alt-button', 'index': alt['FOODNAME']},
                            n_clicks=0,
                            style={'margin': '3px'}
                        )
                        for _, alt in alternatives.head(3).iterrows()
                    ])
                ])
            else:
                recommendation_div = html.P("🌍 This item has high CO₂ emissions, but no good alternative was found.")
        else:
            recommendation_div = html.Div()

    # 4. Update dropdown options based on category
    if selected_recs is not None:
        filtered_df = df_clean[df_clean['recs'] == selected_recs]
        dropdown_options = [{'label': food, 'value': food} for food in sorted(filtered_df['FOODNAME'].unique())]
    else:
        dropdown_options = []

    # 5. Totals
    total_energy = round(sum(item['Energy'] for item in current_data), 2)
    total_calories = round(sum(item['Calories'] for item in current_data), 2)
    total_co2 = round(sum(item['CO2'] for item in current_data), 2)
    total_protein = round(sum(item['protein (g)'] for item in current_data), 2)
    totals_div = html.Div([
        html.H4("Total for Meal"),
        html.P(f"Total Energy: {total_energy} kJ"),
        html.P(f"Total Calories: {total_calories} kCal"),
        html.P(f"Total CO₂: {total_co2} g"),
        html.P(f"Total Protein: {total_protein} g")
    ])

    # 6. Pie chart
    df_meal = pd.DataFrame(current_data)
    if not df_meal.empty:
        df_meal_grouped = df_meal.groupby(['Food'], as_index=False).first()
        df_meal_grouped['recs'] = df_meal_grouped['Food'].map(df_clean.set_index('FOODNAME')['recs'].to_dict())
        fig = px.sunburst(df_meal_grouped, path=['recs', 'Food'], values='Quantity', color='recs')
    else:
        fig = px.pie(names=['None'], values=[1], title='Meal Composition')

    return (
        current_data,
        current_data,
        totals_div,
        None, [],  # active_cell, selected_cells
        fig,
        recommendation_div,
        set_value,  # dropdown value
        None,  # reset quantity
        dropdown_options
    )



@app.callback(
    Output('top-vitamin-plot', 'figure'),
    Input('vitamin-dropdown', 'value')
)
def top_vitamin_plot(vitamin):
    df_vit = df_clean[['FOODNAME', vitamin, 'FOODCLASS', 'CO2/100g', 'energy (kJ)']].dropna()
    df_vit = df_vit.sort_values(by=vitamin, ascending=False).head(15)
    df_vit['Calories (kcal)'] = (df_vit['energy (kJ)'] / 4.184).round(1)
    food_order = df_vit['FOODNAME'].tolist()

    rdl = recommended_daily_levels.get(vitamin, 'N/A')

    fig = px.bar(
        df_vit,
        x='FOODNAME',
        y=vitamin,
        color='FOODCLASS',
        category_orders={'FOODNAME': food_order}
    )

    fig.update_traces(
        text=None,
        customdata=df_vit[['CO2/100g', 'Calories (kcal)']].values,
        hovertemplate=(
            "<b>%{x}</b><br>" +
            "{}: %{{y}}<br>".format(vitamin) +
            "CO₂: %{customdata[0]:.2f} g<br>" +
			"Calories: %{customdata[1]:.1f } kCal<br>" +
            "<extra></extra>"
        )
    )

    fig.update_layout(
        title_text=f"Top 15 Foods Rich in {vitamin} (Recommended daily level: {rdl})",
        title_x=0.5,
        showlegend=True,
        xaxis_title=None,
        yaxis_title=None,
        xaxis_tickangle=-45,
        margin=dict(t=100, b=120)
    )

    return fig

if __name__ == '__main__':
    app.run(debug=True)

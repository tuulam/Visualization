import pandas as pd
import dash
from dash import dcc, html, Input, Output, State, ctx, dash_table, callback_context
import plotly.express as px
from dash import MATCH, ALL
import plotly.colors as pc
import plotly.express as px


# Load and clean your data
df = pd.read_excel('sorted4.xlsx')
df2 = pd.read_excel("environ_grams.xlsx")

carbon_threshold = 50
   

vitamin_rdi_dict = {
    "vitamin C, mg": 90,      # mg
    "vitamin A, µg": 900,     # µg
    "vitamin B12, µg": 2.4,   # µg
	"thiamin, mg": 1.2,		  # mg
	"vitamin B12": 2.4,   # µg
	"vitamin D, µg": 20,      # µg
    "vitamin E, µg": 4,		  # µg
	"vitamin K, µg": 120,	  # µg
	"carotenoids, mg": 2	  # mg
}

env_columns = ['Food emissions of land use',
       'Food emissions of farms', 'Food emissions of animal feed',
       'Food emissions of processing', 'Food emissions of transport',
       'Food emissions of retail', 'Food emissions of packaging',
       'Food emissions of losses']
	   

# Fix all numeric columns that may contain commas instead of dots
exclude = ['id', 'FOODNAME', 'FOODCLASS', 'recs', 'Categories']
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

# Get unique categories from both datasets
categories = df_clean['Categories'].dropna().unique()
entities = df2['Categories'].dropna().unique()


# Generate consistent color map (this one uses Plotly’s 'Plotly' colorway)
category_colors = px.colors.qualitative.Plotly
color_map_categories = {cat: category_colors[i % len(category_colors)] for i, cat in enumerate(categories)}
color_map_entities = {ent: category_colors[i % len(category_colors)] for i, ent in enumerate(entities)}

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
        dcc.Tab(label='Food Details', children=[
            html.H1("Food Nutrition vs. CO2 Emissions", style={'fontWeight': 'bold', 'color': 'green'}),

            html.Div([
                html.P("Select the nutrient you are interested in from the dropdown menu. The higher the nutrient value of the food, the higher it appears in the plot. High carbon dioxide emissions move the food to the right."),
                html.P("Click the Categories legend items to filter food categories. Click a food dot in the plot to see its nutrition details below."),
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
		
	    dcc.Tab(label='Food Supply Chain Emissions', children=[
            html.H1("Environmental Impact of Foods in the Different Stages of the Food Supply Chain", style={'fontWeight': 'bold', 'color': 'red'}),
            html.Label("Select a Stage of the Food Supply Chain:"),
            dcc.Dropdown(
                id='emission-dropdown',
                options=[{'label': v, 'value': v} for v in env_columns],
                value=env_columns[0]
            ),
            dcc.Graph(id='effect-plot'),
			
            html.Div(id='total-impact-output', style={"marginTop": "20px", "fontWeight": "bold", "fontSize": "24px" }),
			
            html.Div(id='sustainability-tip', style={"marginTop": "10px", "color": "green", "fontStyle": "italic"})			
        ]),
		
        
        dcc.Tab(label='Meal Planner', children=[
            html.H1("Meal Planner", style={'fontWeight': 'bold', 'color': 'green'}),
            html.Div([
                html.H2("Nutrition recommendations and food-based dietary guidelines by Finnish Food Authority", style={'fontWeight': 'bold', 'color': 'green'}),
                html.P([
                    "It is recommended to eat at least 500 g of vegetables, fruit, berries, and mushrooms. Of this amount, half should consist of berries and fruit, and the rest vegetables.", html.Br(),
                    "Fish should be eaten two to three times per week, with a variety of species included in rotation.", html.Br(), 
				    "The total weekly intake of red meat and meat products should not exceed 500 grams. A typical cooked portion of fish or meat weighs approximately 100 to 150 grams.", html.Br(),
                    "Legumes should be consumed in amounts of 50 to 100 grams per day.", html.Br(),
				    "A daily intake of 30 grams of nuts and seeds is recommended.", html.Br(),
                    "The recommended daily intake of cereal products—such as cooked whole grain pasta, barley, rice, or whole grain bread—is approximately 600 ml for women and 900 ml for men. At least half of this should come from whole grain sources.", html.Br(),
                    "For environmental reasons, it is not recommended to increase current levels of poultry consumption. Including up to one egg per day can be part of a health-promoting diet."
                    ], style={"padding": "10px","backgroundColor": "#f9f9f9","border": "1px solid #ccc","borderRadius": "10px","marginBottom": "20px"}),
				]),


            html.Label("Select Food Category (Categories):"),
            dcc.Dropdown(
                id='meal-recs-dropdown',
                options=[{'label': c, 'value': c} for c in sorted(df_clean['Categories'].dropna().unique())],
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

            
        ]),
		
        dcc.Tab(label='Top Vitamin Sources', children=[
            html.H1("Top 15 Foods Rich in Selected Vitamin", style={'fontWeight': 'bold', 'color': 'green'}),
            html.Label("Select a Vitamin:"),
            dcc.Dropdown(
                id='vitamin-dropdown',
                options=[{'label': v, 'value': v} for v in vitamin_columns],
                value=vitamin_columns[0]
            ),
            dcc.Graph(id='top-vitamin-plot'),
            html.Div(id='food-recommendation-output', style={'marginTop': '20px', 'fontSize': '18px'})
        ])

    ])
])


@app.callback(
    Output('scatter-plot', 'figure'),
    Input('y-axis-dropdown', 'value')
	
)

def update_scatter(selected_nutrient):
    df_plot = df_clean[['CO2/100g', selected_nutrient, 'FOODNAME', 'Categories']].dropna()
    fig = px.scatter(
        df_plot,
        x='CO2/100g',
        y=selected_nutrient,
        hover_name='FOODNAME',
        color='Categories',
        color_discrete_map=color_map_categories, 
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

    # Format each value: round floats to 2 decimal places
    formatted_record = {k: (f"{v:.2f}" if isinstance(v, (float, int)) else v) for k, v in record.items()}

    return html.Div([
        html.H4(f"Details for {food_name}"),
        html.Ul([html.Li(f"{k}: {v}") for k, v in formatted_record.items()])
    ])

	
@app.callback(
    Output('effect-plot', 'figure'),
    Input('emission-dropdown', 'value') 
)


def update_environmental_charts(selected_effect):
    env_df = df2[['Entity'] + env_columns].dropna()
    fig = px.bar(
        env_df.sort_values(by=selected_effect, ascending=False).round(2).head(15),
        x=selected_effect,
        y='Entity',
        color='Entity',
        color_discrete_map=color_map_categories, 		
        title=f"Top 15 Foods by {selected_effect} Carbon emissions (in CO₂ equivalent grams) per 100 gram of product",
        template='plotly_white'  
    )
    fig.update_layout(
        height=600,
        xaxis_title=None,
        yaxis_title=None,
		showlegend=False)  # This hides the legend
    return fig
	
@app.callback(
    [Output('total-impact-output', 'children'),
     Output('sustainability-tip', 'children')],
    Input('effect-plot', 'clickData')
)
def show_total_impact_with_tip(clickData):
    if clickData:
        food_clicked = clickData['points'][0]['y']
        row = df2[df2['Entity'] == food_clicked]
        numeric_columns = row.iloc[:, 1:].select_dtypes(include='number')
        if not numeric_columns.empty:
            total = numeric_columns.sum(axis=1).values[0]
        else:
            total = 0

        tip = ""
        if total > carbon_threshold:
            tip = f"💡 Consider reducing {food_clicked} or swapping it with a more sustainable alternative."

        return (
            f"Total environmental impact for {food_clicked}: {total:.2f} grams",
            tip
        )
    else:
        return "", ""

    

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
                buttons = []
                for _, alt in alternatives.head(3).iterrows():
                    co2 = round(alt['CO2/100g'], 2)
                    protein = round(alt['protein (g)'], 2)
                    label = f"{alt['FOODNAME']} (CO₂: {co2:.2f}g, Protein: {protein:.2f}g)"
                    buttons.append(
                        html.Button(
                            label,
                            id={'type': 'alt-button', 'index': alt['FOODNAME']},
                            n_clicks=0,
                            style={'margin': '3px'}
                        )
                    )
                recommendation_div = html.Div([
                    html.P("🌍 This item has high CO₂ emissions. Consider these alternatives:"),
                    html.Div(buttons)
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

    # 4. Update dropdown options based on category
    if selected_recs is not None:
        filtered_df = df_clean[df_clean['Categories'] == selected_recs]
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
        df_meal_grouped['Categories'] = df_meal_grouped['Food'].map(df_clean.set_index('FOODNAME')['Categories'].to_dict())
        fig = px.sunburst(df_meal_grouped, path=['Categories', 'Food'], values='Quantity', color='Categories', color_discrete_map=color_map_categories)
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
    [Output('top-vitamin-plot', 'figure'),
     Output('food-recommendation-output', 'children')],
    [Input('vitamin-dropdown', 'value'),
     Input('top-vitamin-plot', 'clickData')]
)


def make_vitamin_plot(vitamin, clickData=None):
    # Ensure that the vitamin is in the dataframe
    df_vit = df_clean[['FOODNAME', vitamin, 'Categories', 'CO2/100g', 'energy (kJ)']].dropna()
    df_vit = df_vit.sort_values(by=vitamin, ascending=False).head(15)
    df_vit['Calories (kcal)'] = (df_vit['energy (kJ)'] / 4.184).round(1)
    food_order = df_vit['FOODNAME'].tolist()

    # Get the recommended daily level (RDI) for the selected vitamin
    rdl = vitamin_rdi_dict.get(vitamin, 'N/A')

    # Create a bar plot with Plotly Express (bars will be horizontal)
    fig = px.bar(
        df_vit,
        x=vitamin,  # This will be on the x-axis (vitamin content)
        y='FOODNAME',
        color='Categories',
        color_discrete_map=color_map_categories, 
        category_orders={'FOODNAME': food_order},
        orientation='h',  # Horizontal bars
        hover_data={'FOODNAME': True, vitamin: True, 'CO2/100g': True, 'Calories (kcal)': True},  # Adding hover data
		custom_data=['FOODNAME', 'CO2/100g', 'Calories (kcal)']  # <-- ADD THIS LINE
    )

    # Update layout to add a title and adjust axis labels
    fig.update_layout(
        title_text=f"Top 15 Foods Rich in {vitamin} (Recommended daily level: {rdl})",
        title_x=0.5,
        height=600,
        showlegend=True,
        xaxis_title=f"{vitamin} per 100g",  # Add a title to the x-axis (vitamin)
        yaxis_title="Food",
        margin=dict(t=100, b=120),
    )

    # Initialize recommendation text if clickData is present
    recommendation_text = "Click on a food bar to see details."  # Default text if no clickData

    if clickData and 'points' in clickData and clickData['points']:
        food_name = clickData['points'][0].get('customdata', [None])[0]
        co2_per_100g = clickData['points'][0].get('customdata', [None, None])[1]
        calories_per_100g = clickData['points'][0].get('customdata', [None, None, None])[2]
        
        # Find RDI for the selected vitamin
        rdi = vitamin_rdi_dict.get(vitamin, 0)  # vitamin_rdi_dict

        if food_name and rdi != 0:
            # Calculate grams needed, calories, and CO₂
            grams_needed = (rdi / clickData['points'][0]['x']) * 100
            total_calories = (calories_per_100g * grams_needed / 100)
            total_co2 = (co2_per_100g * grams_needed / 100)

            recommendation_text = html.Div([
                html.H4(f"{food_name}", style={'color': 'green'}),
                html.P(f"To reach the RDI of {vitamin} ({rdi}), you'd need to eat about {grams_needed:.1f}g of {food_name}."),
                html.P(f"This amount provides about {total_calories:.1f} kcal and emits {total_co2:.2f} g CO₂."),
            ])
        elif clickData:
            recommendation_text = f"No RDI data available for {vitamin} or missing food data."
    else:
        recommendation_text = "Click on a food bar to see details."

    return fig, recommendation_text



	
def update_vitamin_tab(selected_vitamin, clickData):
    # Update the plot based on the selected vitamin
    fig = make_vitamin_plot(selected_vitamin, clickData)

    # Determine food recommendation based on click
    if clickData:
        food_name = clickData['points'][0]['customdata'][0]
        co2_per_100g = clickData['points'][0]['customdata'][1]
        calories_per_100g = clickData['points'][0]['customdata'][2]
        
        # Find RDI for the selected vitamin (adjust as per your data)
        rdi = vitamin_rdi_dict.get(selected_vitamin, 0)

        # Calculate grams needed, calories, and CO₂
        if rdi == 0:
            recommendation_text = f"No RDI data available for {selected_vitamin}."
        else:
            grams_needed = (rdi / clickData['points'][0]['x']) * 100
            total_calories = (calories_per_100g * grams_needed / 100)
            total_co2 = (co2_per_100g * grams_needed / 100)

            recommendation_text = html.Div([
                html.H4(f"{food_name}", style={'color': 'green'}),
                html.P(f"To reach the RDI of {selected_vitamin} ({rdi}), you'd need to eat about {grams_needed:.1f}g of {food_name}."),
                html.P(f"This amount provides about {total_calories:.1f} kcal and emits {total_co2:.2f} g CO₂."),
            ])
    else:
        recommendation_text = "Click on a food bar to see details."

    return fig, recommendation_text


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)

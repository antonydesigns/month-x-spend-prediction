import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import utils_helpers as utils


class Project:
    def __init__(
        self,
        input_paths: dict,
        instance_name="v1",
        allow_instance_rewrite=False,
        warnings=True,
    ):

        self.utils = utils.Utils(instance_name, allow_instance_rewrite, warnings)

        self.input_paths = input_paths
        self.users = pd.DataFrame({})
        self.orders = pd.DataFrame({})
        self.orders_target_sample = pd.DataFrame({})
        self.reports = pd.DataFrame({})

        self.pipeline()

    def pipeline(self):

        # Standard checks to make sure files are put in the right output folder
        self.utils.check_file_paths()

        # Build the appropriate orders list for the target sample
        # Include only users whose first transaction is before 12 months ago
        self.prepare_cleaned_datasets()

        # Create 12 datasets, 1 for each month, into the output folder
        for i in range(12):
            m = i + 1
            print(f"Preparing ML dataset for month {m}")
            self.ml_dataset_preparation(
                self.orders_target_sample, month_number=m, save_to_csv=True
            )

        # Train 12 linear regression models and try different polynomial degrees (1-5)
        # Create a report after looping through all datasets
        print("Training models...")
        reports = self.loop_over_datasets(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [1, 2, 3, 4, 5]
        )
        reports = pd.DataFrame(reports)
        self.utils.to_csv(reports, "report.csv")
        print("Results saved.")
        self.reports = reports

    def prepare_cleaned_datasets(self) -> None:
        """

        Part 1: Preparing the dataset

        """
        print(f"Fetching users list...")
        self.users = pd.read_csv(self.input_paths["segmentation"])

        print(f"Fetching order list...")
        self.orders = pd.read_csv(self.input_paths["orders"])

        print(f"Preparing clean datasets...")
        orders = self.orders.copy()
        orders["Order paid"] = pd.to_datetime(orders["Order paid"])

        # For each user, find their first ever transaction
        customer_created = (
            orders.groupby("User ID")
            .agg({"Order paid": "min"})
            .rename(columns={"Order paid": "Customer created"})
            .reset_index()
        )

        # Create a DF for only targeted users from the segmentation set
        orders = pd.merge(orders, customer_created, "left", "User ID")
        orders_target_sample = pd.merge(self.users, orders, "left", "User ID")

        # Filter out users whose first transaction is less than a year ago
        orders_target_sample = self.ignore_customers_less_than_one_year(
            orders_target_sample
        )

        self.orders_target_sample = orders_target_sample

    def ml_dataset_preparation(
        self, df: pd.DataFrame, month_number: int, save_to_csv: bool = False
    ):
        name = f"M_{month_number}"

        # Calculate total spend in first 12 months
        s = df.copy()
        s["End date"] = s["Customer created"] + pd.DateOffset(months=12)
        s["in"] = (s["End date"] >= s["Order paid"]).astype(int)
        s = s[s["in"] == 1]

        g1 = (
            s.groupby("User ID")
            .agg({"Order item fiat amount (local)": "sum"})
            .rename(
                columns={"Order item fiat amount (local)": "Total spend first year"}
            )
            .reset_index()
        )

        # Calculate total spend in first X months
        s = df.copy()
        s[name] = s["Customer created"] + pd.DateOffset(months=month_number)
        s["in"] = (s[name] >= s["Order paid"]).astype(int)
        s = s[s["in"] == 1]
        g2 = (
            s.groupby("User ID")
            .agg({"Order item fiat amount (local)": "sum"})
            .rename(
                columns={"Order item fiat amount (local)": f"Spend {month_number} mo"}
            )
            .reset_index()
        )

        # Merge g1 and g2 to create feature - target pair

        m = pd.merge(g1, g2, "left", "User ID")

        # Save to CSV optionally

        if save_to_csv:
            self.utils.to_csv(m, f"ML/M_{month_number}.csv")

    def ignore_customers_less_than_one_year(
        self, df: pd.DataFrame, date: str = "2023-08-24"
    ) -> pd.DataFrame:
        t = df.copy()
        t["Cutoff"] = pd.to_datetime(date)
        t["pass"] = (t["Customer created"] <= t["Cutoff"]).astype(int)
        t = t[t["pass"] == 0]
        return t

    def loop_over_datasets(self, months: list, poly_degrees: list) -> list[dict]:
        reports = []
        for month in months:
            df = pd.read_csv(
                utils.relpath(f"{self.utils.instance_name}/ML/M_{month}.csv")
            )
            reports.extend(self.try_different_degrees(df, month, poly_degrees))
        return reports

    def try_different_degrees(
        self, df: pd.DataFrame, month_number: int, degrees: list = [1, 2, 3, 4, 5]
    ) -> list:
        reports = []
        for degree in degrees:
            report = self.train_evaluate(df, degree, month_number)
            reports.append(report)
        return reports

    def train_evaluate(
        self, df: pd.DataFrame, degree, month_number, output_model=False
    ):
        t = df.copy()
        t = t.drop(columns=["User ID"])

        # Transform into 2D array per Preprocessing requirement

        X = t.iloc[:, 1].values.reshape(-1, 1)
        y = t.iloc[:, 0].values

        # Train test split

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.5, random_state=10
        )

        #  Preprocessing

        poly = PolynomialFeatures(degree=degree)
        X_poly_train = poly.fit_transform(X_train)
        X_poly_test = poly.transform(X_test)

        # Train the model
        model = LinearRegression()
        model.fit(X_poly_train, y_train)

        # Predict and evaluate
        y_train_pred = model.predict(X_poly_train)
        y_test_pred = model.predict(X_poly_test)

        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        diff_percent = (test_mse - train_mse) / train_mse * 100

        report = {
            "month_number": month_number,
            "degree": degree,
            "training_rmse": train_mse**0.5,
            "testing_rmse": test_mse**0.5,
            "train/test_mse_diff": diff_percent,
            "train_r2": train_r2,
            "test_r2": test_r2,
            "model_intercept": model.intercept_,
            "coefficients": model.coef_,
        }

        if output_model:
            return model

        return report


paths = utils.input_paths(
    {
        "segmentation": "data/segmentation.csv",
        "orders": "data/All paid NZ orders.csv",
    }
)

p = Project(
    paths,
    allow_instance_rewrite=True,
    warnings=False,
)

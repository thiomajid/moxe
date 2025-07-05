if not isinstance(element, list):
element = [element]

        batch: dict = self.collator.numpy_call(element)

        result = {}
        for key in self.target_columns:
            if key in batch:
                result[key] = jnp.array(batch[key])

        return result


